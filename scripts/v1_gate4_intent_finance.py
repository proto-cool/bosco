"""Gate 4 data, fields `intent` and `finance` (docs/gate4-data-intent-finance.md; plan in
docs/research-gate4/intent-financial.md; decisions 23, 28-32 in docs/DECISIONS-2026-09-25.md).

uv run python scripts/v1_gate4_intent_finance.py fetch            # originals -> data/raw/clean/<source>/
uv run python scripts/v1_gate4_intent_finance.py build            # -> data/cache/v1-gate4-intent-finance/items.json
uv run --with "transformers==4.46.3" --with "sentence-transformers==3.3.1" --with einops \
    python scripts/v1_gate4_intent_finance.py embed               # -> emb.npz (X, L, Z, ZL; family antenna)
uv run python scripts/v1_gate4_intent_finance.py ceilings         # report-only; pool val + held-out DEV only

Rules kept here:
- Only sources verified licence-clean (the doc's table). No LLM-generated text. No NC.
- Personal data: phone numbers, e-mail addresses and long digit runs are scrubbed on read; username, annotator
  and worker fields are never read into text; CFPB "XXXX" masks stay. CFPB state/ZIP/company are dropped.
- Held-out sources (intent: MultiDoGO; finance: CFPB archive, received <= 2022-11-01) are split 20% `dev` /
  80% `test`. `test` is SEALED: it is counted, never scored. `ceilings` refuses to touch it.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import io
import json
import re
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bosco import paths

CLEAN = paths.ROOT / "data" / "raw" / "clean"
OUT = paths.CACHE / "v1-gate4-intent-finance"
FAMILY = paths.ROOT / "service" / "families" / "v1"
CEIL = paths.ROOT / "runs" / "gate4-ceilings"
UA = {"User-Agent": "bosco-research/0.1 (https://bosco.systems)"}
SEED = 20260927

# ---------------------------------------------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------------------------------------------
GH = {  # source dir -> (repo, pinned commit, [paths] or a path prefix filter)
    "polyai": (
        "PolyAI-LDN/task-specific-datasets",
        "57ec275d8078af65b7731c2a98be812d844a6d6b",
        [
            "LICENSE",
            "README.md",
            "banking_data/categories.json",
            "banking_data/train.csv",
            "banking_data/test.csv",
            "nlupp/README.md",
            "nlupp/data/ontology.json",
        ]
        + [f"nlupp/data/{d}/fold{i}.json" for d in ("banking", "hotels") for i in range(20)],
    ),
    "abcd": (
        "asappresearch/abcd",
        "6b8700ce67c6b37b062dd7a60abc76d7ef832a97",
        ["LICENSE", "README.md", "data/abcd_v1.1.json.gz", "data/ontology.json"],
    ),
    "multidogo": (
        "awslabs/multi-domain-goal-oriented-dialogues-dataset",
        "baa30639c4b271f394b81443c842193407cdf26d",
        ["LICENSE.txt", "NOTICE", "README.md"]
        + [
            f"data/paper_splits/splits_annotated_at_turn_level/{d}/{s}.tsv"
            for d in ("airline", "fastfood", "finance", "insurance", "media", "software")
            for s in ("train", "dev", "test")
        ],
    ),
    "snips2017": (
        "sonos/nlu-benchmark",
        "b86ac7f1577868c42158d0dec77db50956046696",  # same commit as snips_built_in_intents (clean-data.md)
        ["LICENSE", "README.md"]
        + [
            f"2017-06-custom-intent-engines/{i}/{k}_{i}{f}.json"
            for i in (
                "AddToPlaylist",
                "BookRestaurant",
                "GetWeather",
                "PlayMusic",
                "RateBook",
                "SearchCreativeWork",
                "SearchScreeningEvent",
            )
            for k, f in (("train", "_full"), ("validate", ""))
        ],
    ),
    "sgd": (
        "google-research-datasets/dstc8-schema-guided-dialogue",
        "e852981ae34990f4358979625854259302feaa78",
        "TREE:^(LICENSE.txt|README.md|(train|dev|test)/.*\\.json)$",
    ),
}
CFPB_ZIPS = [  # received <= 2022-10-31 (Export 4 starts November 2022)
    "CCDB_Export_1_December_2011_through_April_2018.zip",
    "CCDB_Export_2_May_2018_through_April_2021.zip",
    "CCDB_Export_3_May_2021_through_October_2022.zip",
]
CFPB_BASE = "https://files.consumerfinance.gov/f/documents/"
CFPB_KEEP = [
    "Complaint ID",
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
]  # State, ZIP code, Company, Tags etc. are never read
SE_ITEM = "stackexchange_20240402"
SE_FILES = ["license.txt", "readme.txt", "money.stackexchange.com.7z"]
CUTOFF = "2022-11-01"  # pre-LLM text rule (decision 31; CFPB <= 2022-11-01)


def _get(url: str) -> bytes:
    for k in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600).read()
        except Exception as e:  # noqa: BLE001
            if k == 3:
                raise
            print(f"  retry {url}: {e}")
            time.sleep(5 * (k + 1))
    raise RuntimeError


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _record(src: str, entry: dict) -> None:
    f = CLEAN / src / "FETCH.json"
    d = json.load(open(f)) if f.exists() else {"source": src, "files": []}
    d["files"] = [x for x in d["files"] if x["path"] != entry["path"]] + [entry]
    d.update({k: v for k, v in entry.items() if k in ("repo", "commit", "item")})
    json.dump(d, open(f, "w"), indent=1)


def _download(src: str, url: str, rel: str, extra=None) -> Path:
    p = CLEAN / src / rel
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".part")
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=900) as r, open(tmp, "wb") as f:
            while b := r.read(1 << 20):
                f.write(b)
        tmp.rename(p)
    _record(
        src,
        {
            "path": rel,
            "url": url,
            "bytes": p.stat().st_size,
            "sha256": _sha(p),
            "fetched": time.strftime("%Y-%m-%d"),
            **(extra or {}),
        },
    )
    return p


def fetch_github(src: str) -> None:
    repo, sha, want = GH[src]
    head = json.loads(_get(f"https://api.github.com/repos/{repo}/commits?per_page=1"))[0]["sha"]
    if head != sha:
        print(f"  NOTE {repo}: HEAD is {head}, pinned {sha} is used")
    if isinstance(want, str):
        rx = re.compile(want[5:])
        tree = json.loads(_get(f"https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1"))["tree"]
        want = [x["path"] for x in tree if x["type"] == "blob" and rx.match(x["path"])]
    for rel in want:
        _download(src, f"https://raw.githubusercontent.com/{repo}/{sha}/{rel}", rel, {"repo": repo, "commit": sha})
    print(f"{src}: {len(want)} files at {repo}@{sha[:7]}")


def fetch_se() -> None:
    meta = json.loads(_get(f"https://archive.org/metadata/{SE_ITEM}"))
    ia = {f["name"]: f for f in meta["files"]}
    for n in SE_FILES:
        p = _download(
            "money_se",
            f"https://archive.org/download/{SE_ITEM}/{n}",
            n,
            {"item": SE_ITEM, "ia_sha1": ia[n].get("sha1"), "ia_md5": ia[n].get("md5")},
        )
        got = hashlib.sha1(p.read_bytes()).hexdigest()
        assert got == ia[n]["sha1"], (n, got, ia[n]["sha1"])
    print("money_se: archive.org sha1 checks pass")


def extract_se() -> None:
    """Posts.xml -> questions created before CUTOFF: id, date, title, body (HTML stripped), tags.
    OwnerUserId / OwnerDisplayName / LastEditor* are never read."""
    import xml.etree.ElementTree as ET

    import py7zr  # uv run --with py7zr

    d = CLEAN / "money_se"
    out = d / "questions_pre2022-11.jsonl.gz"
    if out.exists():
        return
    if not (d / "extracted" / "Posts.xml").exists():
        with py7zr.SevenZipFile(d / "money.stackexchange.com.7z") as z:
            z.extract(path=d / "extracted", targets=["Posts.xml"])
    n = kept = 0
    # the 2024-04 dump's Posts.xml is UTF-16 LE (BOM) while declaring utf-8: one <row/> per line, parsed alone
    with gzip.open(out, "wt") as w, open(d / "extracted" / "Posts.xml", encoding="utf-16") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("<row "):
                continue
            n += 1
            el = ET.fromstring(line)
            a = el.attrib
            if a.get("PostTypeId") == "1" and a.get("CreationDate", "9") < CUTOFF:
                body = html.unescape(re.sub(r"<[^>]+>", " ", a.get("Body", "")))
                w.write(
                    json.dumps(
                        {
                            "id": int(a["Id"]),
                            "date": a["CreationDate"][:10],
                            "title": html.unescape(a.get("Title", "")),
                            "body": " ".join(body.split()),
                            "tags": re.findall(r"<([^>]+)>", a.get("Tags", ""))
                            or [t for t in a.get("Tags", "").split("|") if t],
                            "closed": bool(a.get("ClosedDate")),
                        }
                    )
                    + "\n"
                )
                kept += 1
    (d / "extracted" / "Posts.xml").unlink()
    print(f"money_se: {n} posts read, {kept} questions before {CUTOFF} -> {out.name}")


def fetch_cfpb() -> None:
    """Each zip: sha256 recorded, then only CFPB_KEEP columns of rows with a narrative, received <= 2022-11-01,
    are written to a gzipped CSV and the zip is deleted."""
    d = CLEAN / "cfpb_archive"
    d.mkdir(parents=True, exist_ok=True)
    for zn in CFPB_ZIPS:
        out = d / (zn[:-4] + ".narratives.csv.gz")
        if out.exists():
            continue
        p = _download("cfpb_archive", CFPB_BASE + zn, zn)
        n = kept = 0
        with zipfile.ZipFile(p) as z:
            members = [m for m in z.namelist() if m.lower().endswith(".csv")]
            with gzip.open(out, "wt", newline="") as w:
                cw = csv.writer(w)
                cw.writerow(CFPB_KEEP)
                for m in members:
                    rd = csv.DictReader(io.TextIOWrapper(z.open(m), encoding="utf-8", errors="replace"))
                    for r in rd:
                        n += 1
                        dt = _date(r["Date received"])
                        if r.get("Consumer complaint narrative", "").strip() and dt <= CUTOFF:
                            r["Date received"] = dt
                            cw.writerow([r[k] for k in CFPB_KEEP])
                            kept += 1
        _record(
            "cfpb_archive",
            {
                "path": out.name,
                "derived_from": zn,
                "rows_read": n,
                "rows_kept": kept,
                "bytes": out.stat().st_size,
                "sha256": _sha(out),
                "members": members,
            },
        )
        p.unlink()
        print(f"cfpb {zn}: {n} rows, {kept} with narrative and <= {CUTOFF}; zip deleted")


def _date(s: str) -> str:
    s = s.strip()
    if re.match(r"\d{4}-\d{2}-\d{2}", s):
        return s[:10]
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", s)
    if m:
        y = int(m[3]) + (2000 if len(m[3]) == 2 else 0)
        return f"{y:04d}-{int(m[1]):02d}-{int(m[2]):02d}"
    return "9999"


def cmd_fetch(a) -> int:
    for src in a.only or ["polyai", "abcd", "multidogo", "snips2017", "sgd", "money_se", "cfpb"]:
        if src in GH:
            fetch_github(src)
        elif src == "money_se":
            fetch_se()
            extract_se()
        elif src == "cfpb":
            fetch_cfpb()
    return 0


# ---------------------------------------------------------------------------------------------------------------
# taxonomies (frozen in docs/gate4-data-intent-finance.md before any model was run)
# ---------------------------------------------------------------------------------------------------------------
INTENT = [
    "book_or_order",
    "cancel",
    "modify",
    "check_status",
    "pay_or_transfer",
    "report_problem",
    "fraud_or_security",
    "account_management",
    "request_item_or_document",
    "find_or_recommend",
    "ask_service_info",
    "ask_fact",
    "how_to",
    "calculate_or_convert",
    "ask_time_date",
    "weather",
    "navigation_traffic",
    "schedule_and_reminders",
    "lists_and_notes",
    "play_media",
    "device_control",
    "communicate",
    "read_news_or_messages",
    "greeting",
    "thanks",
    "affirm",
    "deny",
    "small_talk",
]
FINANCE = [
    "bank_accounts",
    "cards",
    "payments_transfers",
    "credit_reports_scores",
    "consumer_loans",
    "mortgages_home",
    "student_loans",
    "debt_collection_relief",
    "investing",
    "retirement",
    "tax",
    "insurance",
    "fraud_scams",
    "budgeting_saving",
    "business_finance",
]
DESIGN = {"intent": "B", "finance": "B"}  # decision 27: many options -> one learned memory per option
HELD_OUT = {"intent": "multidogo", "finance": "cfpb_archive"}


def _inv(d: dict) -> dict:
    return {lab: cls for cls, labs in d.items() for lab in labs}


# --- intent maps: source label -> class; a label not listed is DROPPED (listed in the doc) ---
CLINC_INTENT = _inv(
    {
        "book_or_order": ["book_flight", "book_hotel", "car_rental", "restaurant_reservation", "uber", "order"],
        "cancel": ["cancel_reservation"],
        "modify": ["credit_limit_change", "insurance_change"],
        "check_status": [
            "flight_status",
            "order_status",
            "application_status",
            "balance",
            "bill_balance",
            "bill_due",
            "transactions",
            "spending_history",
            "pto_balance",
            "pto_request_status",
            "rewards_balance",
            "confirm_reservation",
            "credit_score",
        ],
        "pay_or_transfer": ["pay_bill", "transfer"],
        "report_problem": ["card_declined", "damaged_card", "lost_luggage", "report_lost_card"],
        "fraud_or_security": ["report_fraud", "freeze_account", "account_blocked"],
        "account_management": ["pin_change", "user_name", "change_user_name", "direct_deposit"],
        "request_item_or_document": ["new_card", "replacement_card_duration", "order_checks", "w2"],
        "find_or_recommend": ["travel_suggestion", "meal_suggestion", "restaurant_suggestion"],
        "ask_service_info": [
            "apr",
            "interest_rate",
            "international_fees",
            "credit_limit",
            "carry_on",
            "expiration_date",
            "routing",
            "travel_alert",
            "international_visa",
            "plug_type",
            "what_can_i_ask_you",
            "redeem_rewards",
            "accept_reservations",
            "how_busy",
            "restaurant_reviews",
            "pto_used",
            "payday",
            "income",
            "insurance",
            "taxes",
            "rollover_401k",
            "min_payment",
        ],
        "ask_fact": [
            "definition",
            "fun_fact",
            "calories",
            "nutrition_info",
            "meaning_of_life",
            "vaccines",
            "gas_type",
            "mpg",
            "food_last",
            "ingredients_list",
        ],
        "how_to": [
            "recipe",
            "cook_time",
            "ingredient_substitution",
            "oil_change_how",
            "jump_start",
            "tire_change",
            "improve_credit_score",
        ],
        "calculate_or_convert": [
            "calculator",
            "measurement_conversion",
            "spelling",
            "translate",
            "roll_dice",
            "flip_coin",
            "exchange_rate",
        ],
        "ask_time_date": ["time", "date", "next_holiday", "timezone"],
        "weather": ["weather"],
        "navigation_traffic": ["directions", "distance", "traffic", "current_location", "gas", "find_phone"],
        "schedule_and_reminders": [
            "alarm",
            "timer",
            "reminder",
            "reminder_update",
            "calendar",
            "calendar_update",
            "meeting_schedule",
            "schedule_meeting",
            "schedule_maintenance",
            "pto_request",
        ],
        "lists_and_notes": ["shopping_list", "shopping_list_update", "todo_list", "todo_list_update"],
        "play_media": ["play_music", "next_song", "update_playlist", "what_song"],
        "device_control": [
            "smart_home",
            "change_volume",
            "change_speed",
            "change_accent",
            "change_language",
            "change_ai_name",
            "whisper_mode",
            "reset_settings",
            "sync_device",
            "repeat",
        ],
        "communicate": ["make_call", "text", "share_location"],
        "greeting": ["greeting", "goodbye"],
        "thanks": ["thank_you"],
        "affirm": ["yes"],
        "deny": ["no"],
        "small_talk": [
            "are_you_a_bot",
            "how_old_are_you",
            "tell_joke",
            "what_are_your_hobbies",
            "what_is_your_name",
            "where_are_you_from",
            "who_do_you_work_for",
            "who_made_you",
            "do_you_have_pets",
        ],
    }
)  # dropped: maybe, cancel, last_maintenance, tire_pressure, oil_change_when, travel_notification, oos
MASSIVE_INTENT = _inv(
    {
        "schedule_and_reminders": [
            "alarm_query",
            "alarm_remove",
            "alarm_set",
            "calendar_query",
            "calendar_remove",
            "calendar_set",
        ],
        "lists_and_notes": ["lists_createoradd", "lists_query", "lists_remove"],
        "play_media": ["play_music", "play_radio", "play_podcasts", "play_audiobook", "music_query", "music_settings"],
        "device_control": [
            "iot_cleaning",
            "iot_coffee",
            "iot_hue_lightchange",
            "iot_hue_lightdim",
            "iot_hue_lightoff",
            "iot_hue_lighton",
            "iot_hue_lightup",
            "iot_wemo_off",
            "iot_wemo_on",
            "audio_volume_down",
            "audio_volume_mute",
            "audio_volume_other",
            "audio_volume_up",
        ],
        "communicate": ["email_sendemail", "email_addcontact", "social_post"],
        "read_news_or_messages": ["email_query", "email_querycontact", "social_query", "news_query"],
        "weather": ["weather_query"],
        "ask_time_date": ["datetime_query"],
        "calculate_or_convert": ["datetime_convert", "qa_maths", "qa_currency"],
        "ask_fact": ["qa_factoid", "qa_definition", "qa_stock", "cooking_query"],
        "how_to": ["cooking_recipe"],
        "book_or_order": ["takeaway_order", "transport_ticket", "transport_taxi"],
        "check_status": ["takeaway_query"],
        "navigation_traffic": ["transport_traffic"],
        "find_or_recommend": ["recommendation_events", "recommendation_locations", "recommendation_movies"],
        "greeting": ["general_greet"],
        "small_talk": ["general_joke"],
    }
)  # dropped: general_quirky, music_likeness, music_dislikeness, play_game, transport_query
SNIPS16_INTENT = _inv(
    {
        "communicate": ["ShareCurrentLocation", "ShareETA"],
        "find_or_recommend": ["SearchPlace"],
        "book_or_order": ["BookRestaurant", "RequestRide"],
        "navigation_traffic": ["GetDirections", "GetTrafficInformation"],
        "weather": ["GetWeather"],
    }
)  # dropped: ComparePlaces, GetPlaceDetails
SNIPS17_INTENT = _inv(
    {
        "play_media": ["AddToPlaylist", "PlayMusic"],
        "book_or_order": ["BookRestaurant"],
        "weather": ["GetWeather"],
        "find_or_recommend": ["SearchCreativeWork", "SearchScreeningEvent"],
    }
)  # dropped: RateBook
B77_INTENT = _inv(
    {
        "pay_or_transfer": [
            "transfer_into_account",
            "receiving_money",
            "exchange_via_app",
            "topping_up_by_card",
            "top_up_by_cash_or_cheque",
            "automatic_top_up",
        ],
        "check_status": [
            "pending_cash_withdrawal",
            "pending_top_up",
            "pending_card_payment",
            "pending_transfer",
            "balance_not_updated_after_cheque_or_cash_deposit",
            "balance_not_updated_after_bank_transfer",
            "card_arrival",
            "card_delivery_estimate",
            "transfer_timing",
            "transfer_not_received_by_recipient",
        ],
        "report_problem": [
            "card_not_working",
            "declined_cash_withdrawal",
            "declined_transfer",
            "declined_card_payment",
            "failed_transfer",
            "top_up_failed",
            "top_up_reverted",
            "pin_blocked",
            "card_swallowed",
            "contactless_not_working",
            "virtual_card_not_working",
            "wrong_amount_of_cash_received",
            "lost_or_stolen_card",
            "lost_or_stolen_phone",
            "Refund_not_showing_up",
            "reverted_card_payment?",
            "transaction_charged_twice",
            "extra_charge_on_statement",
            "card_payment_fee_charged",
            "transfer_fee_charged",
            "card_payment_wrong_exchange_rate",
            "wrong_exchange_rate_for_cash_withdrawal",
        ],
        "fraud_or_security": [
            "card_payment_not_recognised",
            "cash_withdrawal_not_recognised",
            "direct_debit_payment_not_recognised",
            "compromised_card",
        ],
        "account_management": [
            "edit_personal_details",
            "change_pin",
            "passcode_forgotten",
            "terminate_account",
            "verify_my_identity",
            "unable_to_verify_identity",
            "why_verify_identity",
            "verify_source_of_funds",
            "activate_my_card",
            "card_linking",
        ],
        "request_item_or_document": [
            "order_physical_card",
            "get_physical_card",
            "getting_spare_card",
            "getting_virtual_card",
            "get_disposable_virtual_card",
            "card_about_to_expire",
        ],
        "cancel": ["cancel_transfer"],
        "ask_service_info": [
            "age_limit",
            "atm_support",
            "card_acceptance",
            "country_support",
            "fiat_currency_support",
            "supported_cards_and_currencies",
            "visa_or_mastercard",
            "disposable_card_limits",
            "top_up_limits",
            "exchange_rate",
            "exchange_charge",
            "cash_withdrawal_charge",
            "top_up_by_bank_transfer_charge",
            "top_up_by_card_charge",
            "apple_pay_or_google_pay",
            "beneficiary_not_allowed",
            "verify_top_up",
        ],
    }
)  # dropped: request_refund
SGD_INTENT = _inv(
    {
        "book_or_order": [
            "ReserveRestaurant",
            "ReserveHotel",
            "ReserveCar",
            "BookHouse",
            "BookAppointment",
            "GetRide",
            "GetTrainTickets",
            "BuyBusTicket",
            "BuyEventTickets",
            "BuyMovieTickets",
            "ScheduleVisit",
            "RentMovie",
            "ReserveOnewayFlight",
            "ReserveRoundtripFlights",
        ],
        "find_or_recommend": [
            "FindRestaurants",
            "SearchHotel",
            "SearchHouse",
            "SearchOnewayFlight",
            "SearchRoundtripFlights",
            "FindApartment",
            "FindHomeByArea",
            "FindMovies",
            "FindEvents",
            "FindProvider",
            "FindAttractions",
            "FindBus",
            "FindTrains",
            "GetCarsAvailable",
            "GetEventDates",
            "GetTimesForMovie",
            "LookupMusic",
            "LookupSong",
            "FindHomeByArea",
        ],
        "check_status": ["CheckBalance"],
        "pay_or_transfer": ["TransferMoney", "MakePayment", "RequestPayment"],
        "schedule_and_reminders": ["AddEvent", "GetEvents", "GetAvailableTime", "AddAlarm", "GetAlarms"],
        "play_media": ["PlaySong", "PlayMedia", "PlayMovie"],
        "weather": ["GetWeather"],
        "communicate": ["ShareLocation"],
    }
)  # any other SGD intent is dropped (printed by build)
ABCD_INTENT = _inv(
    {
        "account_management": [
            "recover_username",
            "recover_password",
            "reset_2fa",
            "manage_change_address",
            "manage_change_name",
            "manage_change_phone",
            "manage_payment_method",
        ],
        "check_status": [
            "status_service_added",
            "status_service_removed",
            "status_shipping_question",
            "status_credit_missing",
            "status_delivery_time",
            "status_payment_method",
            "status_quantity",
            "refund_update",
            "refund_status",
            "shipping_issue/status",
            "status_active",
            "status_due_amount",
            "status_due_date",
            "status_delivery_date",
            "status_questions",
        ],
        "modify": ["manage_upgrade", "manage_downgrade", "shipping_issue/manage", "manage_extension"],
        "book_or_order": ["manage_create"],
        "cancel": ["manage_cancel"],
        "report_problem": [
            "refund_initiate",
            "return_stain",
            "return_color",
            "return_size",
            "shipping_issue/missing",
            "credit_card",
            "shopping_cart",
            "search_results",
            "slow_speed",
            "bad_price_competitor",
            "bad_price_yesterday",
            "out_of_stock_general",
            "out_of_stock_one_item",
            "promo_code_invalid",
            "promo_code_out_of_date",
        ],
        "fraud_or_security": [
            "mistimed_billing_already_returned",
            "mistimed_billing_never_bought",
            "manage_dispute_bill",
        ],
        "pay_or_transfer": ["manage_pay_bill"],
        "ask_service_info": ["single_item_query", "storewide_query", "shipping_issue/cost"],
    }
)  # dropped: status_mystery_fee (status vs unrecognised charge)
MDG_INTENT = _inv(
    {
        "book_or_order": [
            "bookflight",
            "orderpizzaintent",
            "orderdrinkintent",
            "ordersaladintent",
            "orderburgerintent",
            "orderdessertintent",
            "ordersideintent",
            "orderbreakfastintent",
            "startserviceintent",
            "startorder",
        ],
        "cancel": ["cancelserviceintent", "stoporder"],
        "modify": ["changeseatassignment", "changeorder", "upgradeserviceintent", "transferserviceintent"],
        "check_status": [
            "checkbalance",
            "checkclaimstatus",
            "viewbillsintent",
            "checkserverstatus",
            "viewdatausageintent",
        ],
        "pay_or_transfer": ["transfermoney"],
        "report_problem": ["reportlostcard", "reportbrokenphone", "reportbrokensoftware"],
        "fraud_or_security": ["disputecharge"],
        "account_management": ["updateaddress", "openaccount", "closeaccount", "updateaccountinfo"],
        "request_item_or_document": ["replacecard", "getboardingpass", "getproofofinsurance", "orderchecks"],
        "ask_service_info": [
            "getseatinfo",
            "getinformationintent",
            "checkoffereligibility",
            "getroutingnumber",
            "getpromotions",
            "getchannelpackageintent",
        ],
        "greeting": ["openinggreeting", "closinggreeting"],
        "thanks": ["thankyou"],
        "affirm": ["confirmation"],
        "deny": ["rejection"],
    }
)  # dropped: contentonly, outofdomain, softwareupdate, expensereport, providereceipt, any <div> multi-label


def nlupp_intent(v: set) -> str | None:
    """NLU++ is multi-label; only these pre-written single-class patterns are kept."""
    acts = {"acknowledge"}
    if v and v <= {"affirm"} | acts and "affirm" in v:
        return "affirm"
    if v and v <= {"deny"} | acts and "deny" in v:
        return "deny"
    if v == {"greet"} or v == {"end_call"}:
        return "greeting"
    if v == {"thank"}:
        return "thanks"
    verbs = v & {"cancel_close_leave_freeze", "change", "make_open_apply_setup_get_activate"}
    obj = v & {"booking", "appointment"}
    if obj and verbs == {"cancel_close_leave_freeze"}:
        return "cancel"
    if obj and verbs == {"change"}:
        return "modify"
    if obj and verbs == {"make_open_apply_setup_get_activate"}:
        return "book_or_order"
    if v & {"wrong_notworking_notshowing", "lost_stolen"} and not verbs:
        return "report_problem"
    if "balance" in v and v <= {
        "balance",
        "request_info",
        "how_much",
        "account",
        "savings",
        "current",
        "business",
        "credit",
        "debit",
        "existing",
    }:
        return "check_status"
    return None


# --- finance maps ---
CFPB_FIN = {
    "Checking or savings account": "bank_accounts",
    "Bank account or service": "bank_accounts",
    "Credit card or prepaid card": "cards",
    "Credit card": "cards",
    "Prepaid card": "cards",
    "Money transfer, virtual currency, or money service": "payments_transfers",
    "Money transfers": "payments_transfers",
    "Virtual currency": "payments_transfers",
    "Credit reporting, credit repair services, or other personal consumer reports": "credit_reports_scores",
    "Credit reporting": "credit_reports_scores",
    "Payday loan, title loan, or personal loan": "consumer_loans",
    "Consumer Loan": "consumer_loans",
    "Vehicle loan or lease": "consumer_loans",
    "Payday loan": "consumer_loans",
    "Mortgage": "mortgages_home",
    "Student loan": "student_loans",
    "Debt collection": "debt_collection_relief",
}  # dropped: Other financial service
CFPB_FRAUD = {  # (Product, Issue, Sub-issue) -> fraud_scams; exact published field values
    ("Money transfer, virtual currency, or money service", "Fraud or scam", ""),
    ("Money transfers", "Fraud or scam", ""),
    ("Prepaid card", "Fraud or scam", ""),
    ("Credit card or prepaid card", "Getting a credit card", "Card opened as result of identity theft or fraud"),
    ("Checking or savings account", "Opening an account", "Account opened as a result of fraud"),
    ("Credit card", "Identity theft / Fraud / Embezzlement", ""),
}
SE_FIN = _inv(
    {
        "tax": [
            "taxes",
            "income-tax",
            "capital-gains-tax",
            "tax-deduction",
            "irs",
            "state-income-tax",
            "gift-tax",
            "withholding",
            "form-1099",
            "form-w-2",
            "form-w-4",
            "income-tax-refund",
            "tax-credit",
            "property-taxes",
            "sales-tax",
            "vat",
            "hst",
            "payroll-taxes",
            "capital-gain",
            "capital-loss",
            "wash-sale",
        ],
        "investing": [
            "stocks",
            "investing",
            "etf",
            "bonds",
            "options",
            "mutual-funds",
            "index-fund",
            "dividends",
            "trading",
            "stock-markets",
            "stock-analysis",
            "shares",
            "investment-strategies",
            "starting-out-investing",
            "stock-valuation",
            "brokerage",
            "broker",
            "futures",
            "shorting-securities",
            "call-options",
            "put-options",
            "portfolio",
            "day-trading",
            "cryptocurrency",
            "bitcoin",
            "technical-analysis",
            "option-strategies",
            "derivatives",
            "commodities",
            "margin",
            "ipo",
            "online-brokerage",
            "online-trading",
            "limit-order",
            "asset-allocation",
            "value-investing",
            "diversification",
            "government-bonds",
            "fixed-income",
        ],
        "retirement": [
            "401k",
            "ira",
            "roth-ira",
            "retirement",
            "pension",
            "social-security",
            "retirement-plan",
            "roth-401k",
            "rollover",
            "rrsp",
            "annuity",
            "roth-conversion",
        ],
        "insurance": ["insurance", "health-insurance", "life-insurance", "car-insurance"],
        "cards": ["credit-card", "debit-card"],
        "mortgages_home": [
            "mortgage",
            "real-estate",
            "rental-property",
            "refinance",
            "first-time-home-buyer",
            "home-loan",
            "home-ownership",
            "down-payment",
            "mortgage-qualification",
        ],
        "consumer_loans": ["loans", "car-loan", "auto-loan", "personal-loan"],
        "student_loans": ["student-loan"],
        "credit_reports_scores": ["credit-score", "credit-report", "credit-history"],
        "fraud_scams": ["scams", "fraud", "identity-theft"],
        "debt_collection_relief": ["debt", "debt-collection", "collections", "bankruptcy"],
        "bank_accounts": [
            "banking",
            "bank-account",
            "checking-account",
            "savings-account",
            "check",
            "online-banking",
            "deposits",
        ],
        "payments_transfers": [
            "money-transfer",
            "currency",
            "foreign-exchange",
            "paypal",
            "international-transfer",
            "wire-transfer",
            "online-payment",
        ],
        "budgeting_saving": ["budget", "budgeting", "savings", "financial-literacy", "emergency-fund", "expenses"],
        "business_finance": [
            "small-business",
            "self-employment",
            "accounting",
            "llc",
            "limited-liability-company",
            "payroll",
            "s-corporation",
            "bookkeeping",
            "double-entry",
            "contractor",
            "start-up",
        ],
    }
)  # a question is kept only if its tags hit exactly one area; all other tags are ignored
B77_FIN = {}
for _lab, _cls in B77_INTENT.items():
    if _cls == "fraud_or_security":
        B77_FIN[_lab] = "fraud_scams"
    elif (
        "top_up" in _lab
        or "topping_up" in _lab
        or "transfer" in _lab
        or "exchange" in _lab
        or _lab in ("receiving_money", "fiat_currency_support", "beneficiary_not_allowed", "verify_top_up")
    ):
        B77_FIN[_lab] = "payments_transfers"
    elif "card" in _lab or _lab in (
        "pin_blocked",
        "change_pin",
        "contactless_not_working",
        "visa_or_mastercard",
        "apple_pay_or_google_pay",
        "activate_my_card",
    ):
        B77_FIN[_lab] = "cards"
    else:
        B77_FIN[_lab] = "bank_accounts"
B77_FIN["request_refund"] = "bank_accounts"
CLINC_FIN = _inv(
    {
        "bank_accounts": [
            "transactions",
            "balance",
            "freeze_account",
            "account_blocked",
            "interest_rate",
            "routing",
            "order_checks",
            "spending_history",
            "direct_deposit",
        ],
        "payments_transfers": ["transfer", "pay_bill", "exchange_rate"],
        "fraud_scams": ["report_fraud"],
        "cards": [
            "report_lost_card",
            "credit_limit",
            "rewards_balance",
            "new_card",
            "application_status",
            "card_declined",
            "international_fees",
            "apr",
            "redeem_rewards",
            "credit_limit_change",
            "damaged_card",
            "replacement_card_duration",
            "expiration_date",
            "min_payment",
        ],
        "credit_reports_scores": ["credit_score", "improve_credit_score"],
        "tax": ["taxes", "w2"],
        "retirement": ["rollover_401k"],
        "insurance": ["insurance", "insurance_change"],
    }
)  # all other CLINC intents are not finance and are not read; bill_balance, bill_due, income, payday dropped
MDG_FIN = {
    **{
        k: "bank_accounts"
        for k in ("checkbalance", "getroutingnumber", "orderchecks", "openaccount", "closeaccount", "updateaddress")
    },
    **{k: "cards" for k in ("reportlostcard", "replacecard", "checkoffereligibility")},
    "transfermoney": "payments_transfers",
    "disputecharge": "fraud_scams",
    **{k: "insurance" for k in ("getproofofinsurance", "checkclaimstatus", "reportbrokenphone")},
}  # MultiDoGO finance + insurance domains only; dialogue acts, contentonly and outofdomain dropped
SGD_FIN = {
    "CheckBalance": "bank_accounts",
    "TransferMoney": "payments_transfers",
    "MakePayment": "payments_transfers",
    "RequestPayment": "payments_transfers",
}

# ---------------------------------------------------------------------------------------------------------------
# protocol constants
# ---------------------------------------------------------------------------------------------------------------
VAL_FRAC = 0.15  # pool units hashed to val
CAPS = {  # (per source per class, per class over sources)
    "intent": {"train": (400, 800), "val": (60, 150)},
    "finance": {"train": (1000, 1300), "val": (150, 250)},
}
HELD_CAP = 300  # per class, held-out source
DEV_FRAC = 0.20  # held-out units hashed to dev; the rest is the sealed test
MIN_TEST = 50  # held-out classes with fewer sealed-test items are reported as not testable
MIN_WORDS = {"multidogo": 3, "abcd": 5}
NEAR = 0.95

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
PHONE = re.compile(r"(?<![\w/])\+?\d(?:[\s\-().]{0,3}\d){6,}(?![\w/])")  # 7+ digits: phones, card/account numbers
DIGIT = re.compile(r"\d")
YEARS = re.compile(r"^\d{4}\s*-\s*\d{4}$")


def scrub(t: str) -> str:
    t = EMAIL.sub("<email>", t)
    t = PHONE.sub(lambda m: m[0] if YEARS.match(m[0]) else "<number>", t)
    # any other token with 7+ digits (IBANs, member ids) unless it is part of a URL or path
    t = " ".join(
        "<number>" if len(DIGIT.findall(w)) >= 7 and "/" not in w and "http" not in w else w for w in t.split()
    )
    return t


def _h(*parts) -> float:
    return int(hashlib.sha1("|".join(map(str, (SEED, *parts))).encode()).hexdigest()[:12], 16) / 16**12


# ---------------------------------------------------------------------------------------------------------------
# readers: yield (source, unit, text, source_label)
# ---------------------------------------------------------------------------------------------------------------
def r_clinc():
    d = json.load(open(CLEAN / "clinc150" / "data_full.json"))
    for k in ("train", "val", "test"):
        for x, y in d[k]:
            yield "clinc150", x.lower().strip(), x, y


def r_massive():
    for line in open(CLEAN / "massive" / "extracted" / "1.1" / "data" / "en-US.jsonl"):
        r = json.loads(line)  # worker_id is never read
        yield "massive", r["utt"].lower().strip(), r["utt"], r["intent"]


def r_snips16():
    d = json.load(open(CLEAN / "snips_built_in_intents" / "2016-12-built-in-intents_benchmark_data.json"))
    for dom in d["domains"]:
        for it in dom["intents"]:
            for q in it["queries"]:
                yield (
                    "snips_builtin",
                    q["text"].lower().strip(),
                    q["text"],
                    it["benchmark"]["Snips"]["original_intent_name"],
                )


def r_snips17():
    base = CLEAN / "snips2017" / "2017-06-custom-intent-engines"
    for f in sorted(base.glob("*/*.json")):
        if not f.name.startswith(("train_", "validate_")) or (f.name.startswith("train_") and "_full" not in f.name):
            continue
        d = json.load(open(f, encoding="latin-1"))
        for lab, qs in d.items():
            for q in qs:
                t = "".join(x["text"] for x in q["data"])
                yield "snips2017", t.lower().strip(), t, lab


def r_b77():
    for s in ("train", "test"):
        for r in csv.DictReader(open(CLEAN / "polyai" / "banking_data" / f"{s}.csv")):
            yield "banking77", r["text"].lower().strip(), r["text"], r["category"]


def r_nlupp():
    for dom in ("banking", "hotels"):
        for i in range(20):
            for r in json.load(open(CLEAN / "polyai" / "nlupp" / "data" / dom / f"fold{i}.json")):
                yield "nlupp", r["text"].lower().strip(), r["text"], frozenset(r.get("intents", []))


def r_sgd():
    for s in ("train", "dev", "test"):
        for f in sorted((CLEAN / "sgd" / s).glob("dialogues_*.json")):
            for dlg in json.load(open(f)):
                prev = {}
                for t in dlg["turns"]:
                    if t["speaker"] != "USER":
                        continue
                    new = []
                    for fr in t["frames"]:
                        a = fr["state"]["active_intent"]
                        if a != "NONE" and a != prev.get(fr["service"]):
                            new.append(a)
                        prev[fr["service"]] = a
                    if len(new) == 1:
                        yield "sgd", f"{s}/{dlg['dialogue_id']}", t["utterance"], new[0]


def r_abcd():
    d = json.load(gzip.open(CLEAN / "abcd" / "data" / "abcd_v1.1.json.gz"))
    for s, convs in d.items():
        for c in convs:  # the scenario's (generated) personal block is never read
            sub = c["scenario"]["subflow"]
            flow = c["scenario"]["flow"]
            lab = (
                f"shipping_issue/{sub}"
                if flow == "shipping_issue"
                else (flow if flow in ("single_item_query", "storewide_query") else sub)
            )  # data subflows: boots_how_1 ...
            for spk, txt in c["original"]:
                if spk == "customer" and len(txt.split()) >= MIN_WORDS["abcd"]:
                    yield "abcd", f"{s}/{c['convo_id']}", txt, lab
                    break


def r_multidogo(domains=("airline", "fastfood", "finance", "insurance", "media", "software")):
    base = CLEAN / "multidogo" / "data" / "paper_splits" / "splits_annotated_at_turn_level"
    for dom in domains:
        for s in ("train", "dev", "test"):
            for r in csv.DictReader(open(base / dom / f"{s}.tsv"), delimiter="\t"):
                if "<div>" in r["intent"] or len(r["utterance"].split()) < MIN_WORDS["multidogo"]:
                    continue
                yield "multidogo", f"{dom}/{r['conversationId']}", r["utterance"], r["intent"]


def r_money_se():
    for line in gzip.open(CLEAN / "money_se" / "questions_pre2022-11.jsonl.gz", "rt"):
        q = json.loads(line)
        if q["closed"]:
            continue
        areas = {SE_FIN[t] for t in q["tags"] if t in SE_FIN}
        lab = areas.pop() if len(areas) == 1 else ("MULTI" if areas else "NONE")
        yield "money_se", q["id"], f"{q['title']}. {q['body']}", lab


def r_cfpb():
    csv.field_size_limit(sys.maxsize)
    for f in sorted((CLEAN / "cfpb_archive").glob("*.narratives.csv.gz")):
        for r in csv.DictReader(gzip.open(f, "rt")):
            key = (r["Product"], r["Issue"], r["Sub-issue"])
            lab = "FRAUD" if key in CFPB_FRAUD else r["Product"]
            yield "cfpb_archive", r["Complaint ID"], r["Consumer complaint narrative"], lab


POOL = {
    "intent": [
        (r_clinc, CLINC_INTENT),
        (r_massive, MASSIVE_INTENT),
        (r_snips16, SNIPS16_INTENT),
        (r_snips17, SNIPS17_INTENT),
        (r_b77, B77_INTENT),
        (r_nlupp, nlupp_intent),
        (r_sgd, SGD_INTENT),
        (r_abcd, ABCD_INTENT),
    ],
    "finance": [
        (r_money_se, {a: a for a in FINANCE}),
        (r_b77, B77_FIN),
        (r_clinc, CLINC_FIN),
        (lambda: r_multidogo(("finance", "insurance")), MDG_FIN),
        (r_sgd, SGD_FIN),
    ],
}
HELD = {"intent": (r_multidogo, MDG_INTENT), "finance": (r_cfpb, {**CFPB_FIN, "FRAUD": "fraud_scams"})}


def _map(m, lab):
    return m(lab) if callable(m) else m.get(lab)


def _collect(reader, m, dropped: Counter) -> list:
    out = []
    for src, unit, text, lab in reader():
        cls = _map(m, lab)
        if cls is None:
            dropped[(src, str(sorted(lab)) if isinstance(lab, frozenset) else lab)] += 1
            continue
        t = scrub(text)
        if t:
            out.append(
                {
                    "source": src,
                    "unit": str(unit),
                    "text": t,
                    "src_label": lab if isinstance(lab, str) else "+".join(sorted(lab)),
                    "cls": cls,
                }
            )
    return out


def _cap(rows: list, per_src: int, per_cls: int, rng) -> list:
    by = defaultdict(list)
    for r in rows:
        by[(r["source"], r["cls"])].append(r)
    capped = []
    for k in sorted(by):
        v = by[k]
        capped += [v[i] for i in rng.permutation(len(v))[:per_src]]
    by = defaultdict(list)
    for r in capped:
        by[r["cls"]].append(r)
    out = []
    for k in sorted(by):
        v = by[k]
        out += [v[i] for i in rng.permutation(len(v))[:per_cls]]
    return out


def build_field(field: str, rng, report: dict) -> list:
    opts = INTENT if field == "intent" else FINANCE
    dropped = Counter()
    pool = []
    for reader, m in POOL[field]:
        pool += _collect(reader, m, dropped)
    # exact duplicates (normalised text): keep one; texts carrying two classes are dropped entirely
    seen = defaultdict(set)
    for r in pool:
        seen[r["text"].lower()].add(r["cls"])
    conflict = {t for t, c in seen.items() if len(c) > 1}
    uniq, used = [], set()
    for i in rng.permutation(len(pool)):
        r = pool[i]
        k = r["text"].lower()
        if k in conflict or k in used:
            continue
        used.add(k)
        uniq.append(r)
    report["pool_exact_dups_dropped"] = len(pool) - len(uniq) - sum(1 for r in pool if r["text"].lower() in conflict)
    report["pool_conflicting_label_texts_dropped"] = len(conflict)
    report["pool_mapped_before_caps"] = {
        f"{s}:{c}": n for (s, c), n in sorted(Counter((r["source"], r["cls"]) for r in uniq).items())
    }
    tr = [r for r in uniq if _h(field, r["source"], r["unit"]) >= VAL_FRAC]
    va = [r for r in uniq if _h(field, r["source"], r["unit"]) < VAL_FRAC]
    tr = _cap(tr, *CAPS[field]["train"], rng)
    va = _cap(va, *CAPS[field]["val"], rng)
    # held-out: per-class cap, then 20% dev / 80% sealed test by unit
    reader, m = HELD[field]
    held = _collect(reader, m, dropped)
    held_texts = defaultdict(set)
    for r in held:
        held_texts[r["text"].lower()].add(r["cls"])
    held = [r for r in held if len(held_texts[r["text"].lower()]) == 1]
    first = {}
    for i in rng.permutation(len(held)):  # one copy of each exact text
        first.setdefault(held[i]["text"].lower(), held[i])
    held = _cap(list(first.values()), HELD_CAP, HELD_CAP, rng)
    for r in held:
        r["split"] = "dev" if _h(field, "held", r["unit"]) < DEV_FRAC else "test"
    # pool items whose exact text is in the held-out set are dropped (near duplicates are dropped at embed time)
    ht = {r["text"].lower() for r in held}
    n0 = len(tr) + len(va)
    tr = [r for r in tr if r["text"].lower() not in ht]
    va = [r for r in va if r["text"].lower() not in ht]
    report["pool_exact_dups_of_held_out_dropped"] = n0 - len(tr) - len(va)
    report["labels_dropped"] = {f"{s}:{lab}": n for (s, lab), n in sorted(dropped.items())}
    for r in tr:
        r["split"] = "train"
    for r in va:
        r["split"] = "val"
    items = []
    for r in tr + va + held:
        items.append(
            {
                "task": field,
                "split": r["split"],
                "text": r["text"],
                "gold": opts.index(r["cls"]),
                "source": r["source"],
                "unit": r["unit"],
                "src_label": r["src_label"],
            }
        )
    return items


def readable(c: str) -> str:
    return c.replace("_", " ")


def cmd_build(a) -> int:
    rng = np.random.default_rng(SEED)
    items, reports = [], {}
    for field in ("intent", "finance"):
        reports[field] = {}
        items += build_field(field, rng, reports[field])
    tasks = {
        f: {
            "options": [readable(c) for c in (INTENT if f == "intent" else FINANCE)],
            "design": DESIGN[f],
            "held_out_source": HELD_OUT[f],
        }
        for f in ("intent", "finance")
    }
    labels = sorted({o for t in tasks.values() for o in t["options"]})
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"tasks": tasks, "items": items, "labels": labels, "build_report": reports}, open(OUT / "items.json", "w")
    )
    counts_print(items, tasks)
    for f, r in reports.items():
        print(f, {k: v for k, v in r.items() if k not in ("labels_dropped", "pool_mapped_before_caps")})
    return 0


def counts_print(items, tasks):
    for f, t in tasks.items():
        c = Counter(it["split"] for it in items if it["task"] == f)
        print(f"{f}: options {len(t['options'])} design {t['design']} {dict(c)}")
        for s in ("train", "val", "dev", "test"):
            cc = Counter(it["gold"] for it in items if it["task"] == f and it["split"] == s)
            print(f"   {s:5s}", " ".join(f"{t['options'][g][:14]}={cc.get(g, 0)}" for g in range(len(t["options"]))))
        cs = Counter((it["split"], it["source"]) for it in items if it["task"] == f)
        print("   by source:", dict(sorted(cs.items())))


# ---------------------------------------------------------------------------------------------------------------
# embed (pinned nomic encoder, 200-word cut as in scripts/v1_data.py; family antenna as in v1_gate2_data.py)
# ---------------------------------------------------------------------------------------------------------------
def cmd_embed(a) -> int:
    from sentence_transformers import SentenceTransformer

    from bosco import encoders

    t0 = time.time()
    meta = json.load(open(OUT / "items.json"))
    enc = SentenceTransformer(
        encoders.TEXT["id"], revision=encoders.TEXT["revision"], trust_remote_code=True, device=a.device
    )

    def e(xs):
        texts = [encoders.TEXT["prefix"] + " ".join(str(x).split()[:200]) for x in xs]
        return enc.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    X = e([it["text"] for it in meta["items"]])
    L = e(meta["labels"])
    keep = np.ones(len(X), bool)
    near = {}
    for t in meta["tasks"]:
        idx = lambda *ss: np.array([i for i, it in enumerate(meta["items"]) if it["task"] == t and it["split"] in ss])  # noqa: E731
        tr, va, ho = idx("train"), idx("val"), idx("dev", "test")
        # (1) pool items that near-duplicate any held-out item (dev or test) are dropped from the pool
        pool = np.concatenate([tr, va])
        m = np.concatenate([(X[pool[k : k + 4096]] @ X[ho].T).max(1) for k in range(0, len(pool), 4096)]) > NEAR
        keep[pool[m]] = False
        # (2) val items that near-duplicate a (kept) train item are dropped
        trk = tr[keep[tr]]
        m2 = ((X[va] @ X[trk].T).max(1) > NEAR) & keep[va]
        keep[va[m2]] = False
        near[t] = {
            "pool_near_dups_of_held_out_dropped": int(m.sum()),
            "pool_near_dups_of_held_out_by_split": {
                "train": int(keep[tr].size - keep[tr].sum()),
                "val": int(m[len(tr) :].sum()),
            },
            "val_near_dups_of_train_dropped": int(m2.sum()),
        }
    meta["items"] = [it for it, k in zip(meta["items"], keep, strict=True) if k]
    meta["near_dup"] = near
    fam = np.load(FAMILY / "antenna.npz")
    z = lambda A: np.clip(0.5 + ((A - fam["mu"]) @ fam["W"].T) / (2 * float(fam["norm"])), 0, 1).astype(np.float32)  # noqa: E731
    np.savez(OUT / "emb.npz", X=X[keep], L=L, Z=z(X[keep]), ZL=z(L))
    json.dump(meta, open(OUT / "items.json", "w"))
    print(f"embedded {len(X)} items, {len(L)} labels; dropped {int((~keep).sum())}: {near} ({time.time() - t0:.0f}s)")
    print("family antenna applied (service/families/v1)")
    counts_print(meta["items"], meta["tasks"])
    return 0


# ---------------------------------------------------------------------------------------------------------------
# ceilings: report-only, informs the pre-registration. Pool val and held-out DEV only; the sealed test is never
# loaded into any model call (asserted).
# ---------------------------------------------------------------------------------------------------------------
def _bacc(y, p, classes=None):
    classes = sorted(set(y)) if classes is None else classes
    rec = {int(c): float(np.mean(p[y == c] == c)) for c in classes}
    return float(np.mean(list(rec.values()))), rec


def cmd_ceilings(a) -> int:
    from sklearn.linear_model import LogisticRegression

    meta = json.load(open(OUT / "items.json"))
    e = np.load(OUT / "emb.npz")
    items = meta["items"]
    res = {
        "note": "report-only ceilings; pool val and held-out dev only; sealed test never scored",
        "encoder": "nomic-embed-text-v1.5 (bosco.encoders.TEXT), family antenna service/families/v1",
    }
    for t, spec in meta["tasks"].items():
        opts = spec["options"]
        sp = np.array([it["split"] for it in items])
        tk = np.array([it["task"] for it in items]) == t
        y = np.array([it["gold"] for it in items])
        tr, va, dv = (np.where(tk & (sp == s))[0] for s in ("train", "val", "dev"))
        assert not any(items[i]["split"] == "test" for i in np.concatenate([tr, va, dv]))
        n_test = Counter(opts[it["gold"]] for it in items if it["task"] == t and it["split"] == "test")
        dev_cls = sorted(set(y[dv].tolist()))
        testable = [opts[c] for c in dev_cls if n_test.get(opts[c], 0) >= MIN_TEST]
        test_cls = [c for c in dev_cls if opts[c] in testable]
        r = {
            "n": {"train": len(tr), "val": len(va), "dev": len(dv), "test_sealed_count": sum(n_test.values())},
            "held_out_source": spec.get("held_out_source"),
            "held_out_classes_present": [opts[c] for c in dev_cls],
            "held_out_classes_testable": testable,
            "min_test_per_class": MIN_TEST,
            "held_out_sealed_test_per_class": dict(sorted(n_test.items())),
        }
        for rep, M in (("X768", e["X"]), ("Z46", e["Z"])):
            best = None
            for C in (0.1, 1.0, 10.0):
                clf = LogisticRegression(C=C, max_iter=3000)
                clf.fit(M[tr], y[tr])
                b, _ = _bacc(y[va], clf.predict(M[va]))
                r.setdefault(rep, {}).setdefault("val_by_C", {})[str(C)] = round(b, 4)
                if best is None or b > best[0]:
                    best = (b, C, clf)
            b, C, clf = best
            pd = clf.predict(M[dv])
            bd, rec = _bacc(y[dv], pd, dev_cls)
            bt, _ = _bacc(y[dv], pd, test_cls)
            r[rep].update(
                {
                    "C": C,
                    "pool_val_bacc": round(b, 4),
                    "held_out_dev_bacc": round(bd, 4),
                    "held_out_dev_bacc_testable_classes": round(bt, 4),
                    "held_out_dev_recall": {opts[c]: round(v, 3) for c, v in rec.items()},
                    "held_out_dev_confusions_top": [
                        f"{opts[g]} -> {opts[q]}: {n}"
                        for (g, q), n in Counter(
                            (int(g), int(q)) for g, q in zip(y[dv], pd, strict=True) if g != q
                        ).most_common(8)
                    ],
                }
            )
        X = e["X"]
        P = np.stack([X[tr][y[tr] == c].mean(0) for c in range(len(opts))])
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        for nm, idx, cls in (("pool_val", va, None), ("held_out_dev", dv, dev_cls)):
            b, rec = _bacc(y[idx], (X[idx] @ P.T).argmax(1), cls)
            r.setdefault("prototype_X768", {})[f"{nm}_bacc"] = round(b, 4)
            if nm == "held_out_dev":
                r["prototype_X768"]["held_out_dev_recall"] = {opts[c]: round(v, 3) for c, v in rec.items()}
        res[t] = r
        print(
            t,
            json.dumps({k: v for k, v in r.items() if k not in ("held_out_sealed_test_per_class",)}, indent=None)[
                :3000
            ],
        )
    CEIL.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(CEIL / "intent-finance.json", "w"), indent=1)
    return 0


# ---------------------------------------------------------------------------------------------------------------
# provenance manifest (rows per source; sha256 of every fetched file from data/raw/clean/<source>/FETCH.json)
# ---------------------------------------------------------------------------------------------------------------
MANIFEST = paths.ROOT / "docs" / "gate4-data-manifest-intent-finance.json"
SOURCES = {
    "clinc150": dict(
        name="CLINC150 (oos-eval data_full.json)",
        url="https://github.com/clinc/oos-eval",
        version="git 828f8093932c8fe6ca7936c3d2e52903b1c523de",
        licence="CC BY 3.0",
        licence_where="repo LICENSE (verified 2026-09-25, docs/clean-data.md; data/raw/clean/MANIFEST.json)",
        personal_data="none",
        fetch="already fetched (data/raw/clean/MANIFEST.json)",
    ),
    "massive": dict(
        name="MASSIVE 1.1 en-US",
        url="https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz",
        version="1.1",
        licence="CC BY 4.0",
        licence_where="tarball LICENSE/NOTICE (docs/clean-data.md)",
        personal_data="worker_id never read",
        fetch="already fetched (data/raw/clean/MANIFEST.json)",
    ),
    "snips_builtin": dict(
        name="SNIPS built-in intents 2016-12",
        url="https://github.com/sonos/nlu-benchmark",
        version="git b86ac7f1577868c42158d0dec77db50956046696",
        licence="CC0 1.0",
        licence_where="repo LICENSE (docs/clean-data.md)",
        personal_data="none",
        fetch="already fetched (data/raw/clean/MANIFEST.json)",
    ),
    "snips2017": dict(
        name="SNIPS 2017-06 custom-intent engines",
        url="https://github.com/sonos/nlu-benchmark",
        version="git b86ac7f1577868c42158d0dec77db50956046696",
        licence="CC0 1.0",
        licence_where="repo LICENSE at the pinned commit, re-read 2026-09-27",
        personal_data="none",
        dir="snips2017",
    ),
    "banking77": dict(
        name="BANKING77",
        url="https://github.com/PolyAI-LDN/task-specific-datasets",
        version="git 57ec275d8078af65b7731c2a98be812d844a6d6b",
        licence="CC BY 4.0",
        licence_where="repo LICENSE (CC BY 4.0 legal code) + README §License, re-read 2026-09-27",
        personal_data="none",
        dir="polyai",
    ),
    "nlupp": dict(
        name="NLU++ (banking, hotels)",
        url="https://github.com/PolyAI-LDN/task-specific-datasets",
        version="git 57ec275d8078af65b7731c2a98be812d844a6d6b",
        licence="CC BY 4.0",
        licence_where="same repo LICENSE, re-read 2026-09-27",
        personal_data="none",
        dir="polyai",
    ),
    "sgd": dict(
        name="Schema-Guided Dialogue (SGD)",
        url="https://github.com/google-research-datasets/dstc8-schema-guided-dialogue",
        version="git e852981ae34990f4358979625854259302feaa78",
        licence="CC BY-SA 4.0",
        licence_where="README §License + LICENSE.txt, re-read 2026-09-27",
        personal_data="none (fictional entities)",
        dir="sgd",
        share_alike=True,
    ),
    "abcd": dict(
        name="ABCD v1.1",
        url="https://github.com/asappresearch/abcd",
        version="git 6b8700ce67c6b37b062dd7a60abc76d7ef832a97",
        licence="MIT",
        licence_where="repo LICENSE, re-read 2026-09-27",
        personal_data="generated scenario personal block never read; text scrubbed",
        dir="abcd",
    ),
    "multidogo": dict(
        name="MultiDoGO (turn-level paper splits)",
        url="https://github.com/awslabs/multi-domain-goal-oriented-dialogues-dataset",
        version="git baa30639c4b271f394b81443c842193407cdf26d",
        licence="CDLA-Permissive-1.0",
        licence_where="LICENSE.txt first line + README §License, re-read 2026-09-27",
        personal_data="role-played; text scrubbed",
        dir="multidogo",
    ),
    "money_se": dict(
        name="Personal Finance & Money Stack Exchange (2024-04-02 dump)",
        url="https://archive.org/details/stackexchange_20240402",
        version="archive.org item stackexchange_20240402, money.stackexchange.com.7z; questions created < 2022-11-01",
        licence="CC BY-SA (2.5/3.0/4.0 by post date)",
        licence_where="dump license.txt (cc-wiki, CC BY-SA 3.0 link, attribution required), read 2026-09-27; "
        "CAUTION accepted by Nick, decision 31 (docs/DECISIONS-2026-09-25.md)",
        personal_data="Owner/Editor fields never read; attribution by question id "
        "(https://money.stackexchange.com/q/<unit>); text scrubbed",
        dir="money_se",
        share_alike=True,
        card_note="newer SE download terms say 'not training a large language model'; Bosco is not one",
    ),
    "cfpb_archive": dict(
        name="CFPB Consumer Complaint Database Narratives Archive, Exports 1-3",
        url="https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/"
        "cfpb-consumer-complaint-database-narratives-archive/",
        version="Exports 1-3 (Dec 2011 - Oct 2022), Last-Modified 2026-09-14; rows received <= 2022-10-31",
        licence="public domain for FOIA purposes (no licence field)",
        licence_where="CFPB newsroom 2026-08-14 'The CFPB to Cease Discretionary Publication of Complaint "
        "Narratives...' and consumerfinance.gov website privacy policy, both read 2026-09-27; CAUTION -> USE "
        "(docs/free-datasets.md)",
        personal_data="consumer-written, opt-in, CFPB-scrubbed (XXXX kept); state/ZIP/company/tags never kept; "
        "zips deleted after extracting 7 columns; text scrubbed",
        dir="cfpb_archive",
    ),
}


def cmd_manifest(a) -> int:
    meta = json.load(open(OUT / "items.json"))
    used = Counter((it["task"], it["source"], it["split"]) for it in meta["items"])
    rows = []
    for src, m in SOURCES.items():
        fetch = CLEAN / m.get("dir", src) / "FETCH.json"
        if fetch.exists():
            files = json.load(open(fetch))["files"]
        else:  # fetched earlier: hashes from data/raw/clean/MANIFEST.json
            nm = {"snips_builtin": "snips_built_in_intents"}.get(src, src)
            ds = [x for x in json.load(open(CLEAN / "MANIFEST.json"))["datasets"] if x["name"] == nm]
            files = ds[0]["files"] if ds else []
        if src in ("banking77", "nlupp"):
            keep = "banking_data/" if src == "banking77" else "nlupp/"
            files = [f for f in files if f["path"].startswith(keep) or f["path"] in ("LICENSE", "README.md")]
        for field in ("intent", "finance"):
            n = {s: used[(field, src, s)] for s in ("train", "val", "dev", "test") if used[(field, src, s)]}
            if not n:
                continue
            rows.append(
                {
                    "row": f"{field}/{src}",
                    "field": field,
                    "source": src,
                    "role": "held-out" if HELD_OUT[field] == src else "pool",
                    **{k: v for k, v in m.items() if k not in ("dir",)},
                    "rows_used": n,
                    "files": [
                        {
                            k: f[k]
                            for k in (
                                "path",
                                "url",
                                "bytes",
                                "sha256",
                                "fetched",
                                "derived_from",
                                "rows_read",
                                "rows_kept",
                            )
                            if k in f
                        }
                        for f in files
                    ],
                }
            )
    json.dump(
        {
            "generated": time.strftime("%Y-%m-%d"),
            "builder": "scripts/v1_gate4_intent_finance.py",
            "doc": "docs/gate4-data-intent-finance.md",
            "rows": rows,
        },
        open(MANIFEST, "w"),
        indent=1,
    )
    print(f"{len(rows)} rows -> {MANIFEST.relative_to(paths.ROOT)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--only", nargs="*")
    p.set_defaults(fn=cmd_fetch)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    p = sub.add_parser("embed")
    p.add_argument("--device", default="mps")
    p.set_defaults(fn=cmd_embed)
    sub.add_parser("ceilings").set_defaults(fn=cmd_ceilings)
    sub.add_parser("manifest").set_defaults(fn=cmd_manifest)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
