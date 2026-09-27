# Gate 4 data research: intent and finance specialists, checked 2026-09-27

Research only. Nothing was trained, nothing was committed, and nothing was fetched into
`data/`. To read labels and headers I did two things:
- fetched the MultiDoGO dev splits (about 1.7 MB) and the SGD `schema.json` files (128 kB) into `/tmp`;
- read the CFPB archive and MTOP/TOP zips by HTTP range request (zip directory, licence and README members, and the first 64 kB of one CSV).

User-Agent was `bosco-research/0.1 (https://bosco.systems)`. None of this is legal advice.

This builds on `docs/free-datasets.md` and `docs/clean-data.md`. For CLINC150 (CC BY 3.0),
MASSIVE (CC BY 4.0), SNIPS built-in (CC0), Wikipedia/Wikidata (`docs/wiki-data.md`) and
the CFPB row, those files already hold the verification. It is not repeated here, only
extended.

Verdicts: **USE** means the licence was verified at the source and provenance is clean.
**CAUTION** means usable only if the stated risk is accepted. **NO** means do not use.

---

## 1. Intent: "what does the person want"

### 1.1 Candidates

| name | maker | URL | licence (where verified) | collection | personal data | size | labels | verdict |
|---|---|---|---|---|---|---|---|---|
| CLINC150 | Clinc / Larson et al. 2019 | github.com/clinc/oos-eval | CC BY 3.0 (repo LICENSE; already in `clean-data.md`, commit `828f809`) | Crowd-written queries per intent | none | 23,700 | 150 intents in 10 domains (banking, credit cards, travel, auto, home, kitchen, work, utility, small talk, meta) + out-of-scope | **USE** (already fetched) |
| MASSIVE en-US | Amazon | amazon-massive-dataset-1.1 | CC BY 4.0 (tarball LICENSE; `clean-data.md`) | SLURP text, localised by MTurk workers | `worker_id` column: drop it | 16.5k | 60 intents / 18 scenarios (voice assistant) | **USE** (already fetched). **Same text family as HWU64**, see leakage |
| HWU64 (NLU-Evaluation-Data) | Liu et al. 2019, Heriot-Watt | github.com/xliuhw/NLU-Evaluation-Data | **CC BY 4.0** (repo LICENSE and README §License; commit `f6071b4`) | Crowd answers to "what would you say to a home robot" prompts | `userid` column | 25,716 annotated | 64 intents / 21 scenarios | **USE licence-wise, but do not add it**. SLURP is "a spoken version of" Liu et al. 2019, and MASSIVE is SLURP localised. Its text is therefore largely the same as MASSIVE en-US. Treat HWU64 and MASSIVE as **one source**. Never train on one and test on the other |
| BANKING77 | PolyAI (Casanueva et al. 2020) | github.com/PolyAI-LDN/task-specific-datasets | **CC BY 4.0** (repo LICENSE, commit `57ec275`; README lists "Banking") | Online-banking customer queries (paper; not synthetic) | none seen | 10,003 train / 3,080 test | 77 fine-grained intents (cards, top-ups, transfers, identity checks) | **USE**. The verification is now at the publisher's repo, not only the HF card |
| NLU++ | PolyAI (Casanueva et al. 2022) | same repo, `nlupp/` | CC BY 4.0 (same repo LICENSE) | Written by dialogue experts (paper; not rechecked) | none | 3,080 (banking 2,071, hotels 1,009) | **Multi-label, compositional** intents (verbs like make/change/cancel/why/how_much/not_working + objects like booking/transfer/card). 62 intents | **USE**. Its verb-type intents fit a domain-general taxonomy well. Possible near-duplicates of BANKING77 (same maker): check them |
| SNIPS built-in intents | Sonos/Snips | github.com/sonos/nlu-benchmark `2016-12-built-in-intents` | CC0 (repo LICENSE, commit `b86ac7f`) | 328 queries, authorship unknown | none | 328 | 10 intents | **USE** (already fetched). Tiny |
| SNIPS 2017 custom-intent benchmark | Sonos/Snips | same repo, `2017-06-custom-intent-engines/` | CC0 (same repo LICENSE) | Crowdsourced (blog; unverified) | none | ~2k per intent (unverified) | 7: AddToPlaylist, BookRestaurant, GetWeather, PlayMusic, RateBook, SearchCreativeWork, SearchScreeningEvent | **USE**. Assistant domain only |
| Schema-Guided Dialogue (SGD) | Google (Rastogi et al. 2020) | github.com/google-research-datasets/dstc8-schema-guided-dialogue | **CC BY-SA 4.0** (repo LICENSE.txt and README §License; commit `e852981`) | Dialogue simulator produces outlines, then paid crowd-workers **paraphrase** them into natural text (README). Pre-LLM (2019) | none (entities are fictional/KB) | ~20k dialogues (paper) | Per-turn `active_intent` per service. Train: 26 services, e.g. Banks CheckBalance/TransferMoney, Flights Search/Reserve, Hotels, RentalCars, Restaurants, RideSharing GetRide, Services BookAppointment, Calendar, Music, Media, Weather, Homes. Test adds Alarm, Messaging ShareLocation, Payment Make/RequestPayment, Trains | **USE** (share-alike: attribute; derived data stays BY-SA). Take the user turn where `active_intent` first changes from NONE |
| MultiWOZ 2.2 | Cambridge (Budzianowski et al.; 2.2 by Zang et al.) | github.com/budzianowski/multiwoz | **MIT** (repo LICENSE, commit `fe0c8e6`) | Wizard-of-Oz, both sides crowd-workers (AMT) | none (fictional Cambridge KB) | ~10k dialogues | 2.2 `active_intent`: find/book restaurant, hotel, train; find attraction, taxi, police, hospital | **USE**. Narrow intents (find vs book) |
| Taskmaster-1/2/3/4 | Google (Byrne et al.) | github.com/google-research-datasets/Taskmaster | **CC BY 4.0**. Each TM-*/README.md: "made available under the Creative Commons Attribution 4.0 License" (commit `d92cb6a`) | TM-1: WOz (crowd user, call-centre operator) and crowd "self-dialogs". TM-2: spoken WOz. TM-3/4: crowd self-dialogs | none expected | 55k+ dialogues | **Domain only**: pizza, auto repair, rides, movie tickets, coffee, restaurants, flights, hotels, music, sports. **No utterance-level intent** | **USE** licence-wise, **low value**: only a first-substantive-user-turn → book_or_order/find_or_recommend weak label. Skip for gate 4 |
| MultiDoGO | Amazon (Peskov et al., EMNLP 2019) | github.com/awslabs/multi-domain-goal-oriented-dialogues-dataset | **CDLA-Permissive-1.0** (repo LICENSE.txt, first line "Community Data License Agreement – Permissive – Version 1.0"; README §License; commit `baa3063`) | "Wizard-of-Oz approach wherein a crowd-sourced worker (the 'customer') is paired with a trained annotator (the 'agent')" (ACL abstract) | role-played; may contain made-up names/numbers (not checked) | 81k dialogues, 54k annotated; turn-level dev ≈ 2k customer turns per domain, train ≈ 7× | Per customer turn: domain intents (airline: bookflight, changeseatassignment, getboardingpass, getseatinfo; fastfood: order*intent; finance: checkbalance, transfermoney, reportlostcard, replacecard, disputecharge, updateaddress, getroutingnumber, orderchecks, open/closeaccount, checkoffereligibility; insurance: getproofofinsurance, reportbrokenphone, checkclaimstatus; media: start/transfer/upgrade/cancelservice, viewbills, getinformation; software: reportbrokensoftware, softwareupdate, startorder, changeorder, checkserverstatus, expensereport, getpromotions) **plus dialogue acts**: openinggreeting, closinggreeting, thankyou, confirmation, rejection, outofdomain, contentonly. Multi-labels joined by `<div>` | **USE**. The broadest customer-support source found: 6 domains, and a different collection method from all the query-list sets |
| ABCD | ASAPP (Chen et al., NAACL 2021) | github.com/asappresearch/abcd | **MIT** (repo LICENSE, commit `6b8700c`) | Human-human customer-service chats with crowd-workers on both sides, following a scenario (README) | scenario names/addresses are generated, not real (paper; not rechecked) | >10k dialogues | 10 flows / 55 subflows (the ground-truth user intent): account_access, manage_account, order_issue, product_defect, purchase_dispute, shipping_issue, single_item_query, storewide_query, subscription_inquiry, troubleshoot_site (`data/ontology.json`) | **USE**. E-commerce support. Label = the dialogue's subflow, so take only the first customer utterance(s) that state the problem |
| STAR | Rasa (Mosig et al. 2020) | github.com/RasaHQ/STAR | **MIT** (repo LICENSE.txt, commit `3058975`) | WOz on AMT (anonymised worker ids) | anonymised worker ids | ~6.6k dialogues (paper; unverified) | Dialogue task per domain (bank, ride, hotel, doctor, meeting, weather, trivia, spaceship…). No clean user-intent per utterance | **USE** licence-wise, **low value** |
| TOP | Facebook (Gupta et al. 2018) | download.pytorch.org/data/semanticparsingdialog.zip (fb.me/semanticparsingdialog) | **CC BY-SA** (zip README: "Provided under the CC-BY-SA license"; version not stated) | Crowd-written | none | 44,783 | 25 intents, navigation + events (IN:GET_DIRECTIONS, IN:GET_EVENT…) | **USE** (BY-SA). Narrow |
| MTOP (English) | Facebook (Li et al., EACL 2021) | dl.fbaipublicfiles.com/mtop/mtop.zip (8.6 MB) | **CC BY-SA 4.0** (`mtop/LICENSE.txt` inside the zip, read by range request) | Crowd-written English, translated for 5 other languages | none | en ≈ 22k (paper; unverified) | 117 intents / 11 domains: alarm, calling, event, messaging, music, news, people, recipes, reminder, timer, weather | **USE** (BY-SA). Assistant domain, **independent authorship from MASSIVE**. A good alternative held-out source for the assistant intents |
| Braun et al. NLU corpora | TU Munich (sebis) | github.com/sebischair/NLU-Evaluation-Corpora | **CC BY-SA 3.0** (README §License) | AskUbuntu + WebApps SE questions; Telegram bot questions (Munich transit) | SE usernames absent; Telegram = real users | ~500 total | IT-help intents: MakeUpdate, SetupPrinter, ShutdownComputer, SoftwareRecommendation, ChangePassword, DeleteAccount, ExportData, FilterSpam, SyncAccounts… | **USE** (SE parts, BY-SA), **CAUTION** (Telegram part: real users, no consent statement). Tiny, but the only IT-help intent set found |
| HINT3 | Haptik | github.com/hellohaptik/HINT3 | **ODbL 1.0 + DbCL 1.0** (repo LICENSE.md) | Queries from real users of 3 Haptik client bots (SOFMattress, Curekart, Powerplay11) (paper; unverified) | real user queries | ~2–4k per bot | bot-specific intents | **CAUTION**. ODbL is not on our list, though it allows commercial use and "produced works" (models) need only a notice. The real-user text, with no consent statement, is the larger issue. Not needed |
| ACID | American Family Insurance | github.com/AmFamMLTeam/ACID | **none**: GitHub reports no licence; README has no licence text | "collected from past interaction of customers with our service representatives" (README) | real customer text | 22k, 175 intents | insurance intents | **NO**. No licence, real customer data |
| ATIS | DARPA / LDC | catalog.ldc.upenn.edu/LDC93S4B etc. | **LDC licence**: non-members "may use LDC data for noncommercial linguistic research and education only" (LDC licensing page) | 1990s spoken air-travel queries | — | ~5k | 17–26 flight intents | **NO**. The widely mirrored JSON/CSV copies carry no licence of their own |
| Bitext customer-support | Bitext | HF bitext/Bitext-customer-support-llm-chatbot-training-dataset | CDLA-Sharing-1.0 (HF card; `free-datasets.md`) | **Synthetic** (NLG from seed texts) | none | 26,872 | 27 support intents | **Synthetic, listed separately**. At most a flagged training supplement, never test |

Not pursued: Frames (Maluuba/Microsoft research terms), DSTC2 (no clear licence),
Facebook multilingual task-oriented set (`multilingual_task_oriented_dialog_slotfilling.zip`
returns HTTP 403).

### 1.2 Proposed taxonomy: 28 domain-general intents

The labels name **the goal type, not the domain**, so that a held-out source from other
domains can still be scored. Assistant-only intents (weather, media, devices) are kept
because the assistant sources dominate the volume.

| # | intent | meaning |
|---|---|---|
| 1 | book_or_order | reserve, book, buy or order something: tables, rooms, flights, cars, rides, tickets, appointments, food, a service |
| 2 | cancel | cancel an existing booking, order, subscription, service or transfer |
| 3 | modify | change an existing booking, order, plan, seat or service (upgrade/downgrade included) |
| 4 | check_status | status of an order, booking, flight, claim, application, delivery, or balance/bill/transactions |
| 5 | pay_or_transfer | pay a bill, send/request/receive money, top up |
| 6 | report_problem | something failed or broke: declined, not working, lost, damaged, missing, wrong amount, broken device |
| 7 | fraud_or_security | unrecognised charge, compromised card, dispute of a charge, report fraud, freeze account |
| 8 | account_management | open/close account, update personal details, password/PIN/2FA, identity verification |
| 9 | request_item_or_document | new or replacement card, boarding pass, proof of insurance, checks, statements, tax forms |
| 10 | find_or_recommend | search or ask for suggestions: restaurants, hotels, flights, movies, events, providers, homes, attractions |
| 11 | ask_service_info | rules, fees, limits, eligibility, how a product or service works, promotions, opening hours |
| 12 | ask_fact | general-knowledge, definition, trivia, nutrition, stock price, news-free facts |
| 13 | how_to | instructions: recipes, car maintenance, software/printer setup |
| 14 | calculate_or_convert | maths, currency, units, time zones, spelling |
| 15 | ask_time_date | time, date, holidays |
| 16 | weather | weather queries |
| 17 | navigation_traffic | directions, distance, traffic, public transport times, current location |
| 18 | schedule_and_reminders | alarms, timers, reminders, calendar events, meetings (set, query or remove) |
| 19 | lists_and_notes | shopping lists, to-do lists |
| 20 | play_media | play or control music, podcasts, radio, audiobooks, films; playlists |
| 21 | device_control | smart-home devices, volume, assistant settings |
| 22 | communicate | call, text, email, post, share location/ETA, add contact |
| 23 | read_news_or_messages | news, email and social queries |
| 24 | greeting | hello and goodbye |
| 25 | thanks | thank you |
| 26 | affirm | yes, confirmation |
| 27 | deny | no, rejection |
| 28 | small_talk | questions about the assistant, jokes, chit-chat |
| (+) | out_of_scope | optional 29th class, from CLINC `oos` and MultiDoGO `outofdomain` |

### 1.3 Mapping plan (write it as a YAML before any data is looked at in bulk)

- **CLINC150.**
  - book_or_order: book_flight, book_hotel, car_rental, restaurant_reservation, uber, order, accept_reservations. uber is a ride booking, so it goes here rather than navigation.
  - cancel: cancel_reservation, cancel. `cancel` alone is ambiguous in CLINC: it means "cancel that" (the assistant command). **Drop `cancel`.**
  - modify: calendar_update, reminder_update, todo_list_update, shopping_list_update go to their target classes (18/19). credit_limit_change, change_* settings → device_control. pin_change → account_management.
  - check_status: flight_status, order_status, application_status, balance, bill_balance, bill_due, transactions, spending_history, pto_balance, pto_request_status, rewards_balance, confirm_reservation.
  - pay_or_transfer: pay_bill, transfer.
  - report_problem: card_declined, damaged_card, lost_luggage, report_lost_card.
  - fraud_or_security: report_fraud, freeze_account, account_blocked.
  - account_management: pin_change, user_name, change_user_name, direct_deposit.
  - request_item_or_document: new_card, replacement_card_duration, order_checks, w2.
  - ask_service_info: apr, interest_rate, international_fees, credit_limit, carry_on, expiration_date, routing, travel_alert, international_visa, plug_type, what_can_i_ask_you, redeem_rewards, accept_reservations, how_busy, restaurant_reviews, pto_used, payday, income, insurance, taxes, rollover_401k, min_payment.
  - ask_fact: definition, fun_fact, calories, nutrition_info, meaning_of_life, vaccines, gas_type, mpg, food_last, ingredients_list, exchange_rate. exchange_rate is a fact lookup, not a conversion; if in doubt, drop it.
  - how_to: recipe, cook_time, ingredient_substitution, oil_change_how, jump_start, tire_change.
  - calculate_or_convert: calculator, measurement_conversion, spelling, translate, roll_dice, flip_coin.
  - ask_time_date: time, date, next_holiday, timezone.
  - weather: weather.
  - navigation_traffic: directions, distance, traffic, current_location, gas, find_phone.
  - schedule_and_reminders: alarm, timer, reminder, reminder_update, calendar, calendar_update, meeting_schedule, schedule_meeting, schedule_maintenance, pto_request.
  - lists_and_notes: shopping_list(_update), todo_list(_update).
  - play_media: play_music, next_song, update_playlist, what_song.
  - device_control: smart_home, change_volume, change_speed, change_accent, change_language, change_ai_name, whisper_mode, reset_settings, sync_device, repeat.
  - communicate: make_call, text, share_location.
  - greeting: greeting, goodbye.
  - thanks: thank_you.
  - affirm: yes.
  - deny: no.
  - small_talk: are_you_a_bot, how_old_are_you, tell_joke, what_are_your_hobbies, what_is_your_name, where_are_you_from, who_do_you_work_for, who_made_you, do_you_have_pets.
  - find_or_recommend: travel_suggestion, meal_suggestion, restaurant_suggestion.
  - improve_credit_score → how_to. credit_score → check_status. text → communicate.
  - **Drop:** maybe, cancel, last_maintenance, tire_pressure, oil_change_when, travel_notification.
  - A few names appear in two lists above (accept_reservations; calendar_update and reminder_update; exchange_rate). **The pre-registration must pick one.** This draft is a starting point to be frozen, not the frozen file.
- **MASSIVE.**
  - alarm_\*, calendar_\*, reminder → schedule_and_reminders.
  - lists_\* → lists_and_notes.
  - play_\*, music_\* → play_media.
  - iot_\*, audio_volume_\* → device_control.
  - email_sendemail, email_addcontact, social_post → communicate.
  - email_query, email_querycontact, social_query, news_query → read_news_or_messages.
  - weather_query → weather.
  - datetime_query → ask_time_date.
  - datetime_convert, qa_maths, qa_currency → calculate_or_convert.
  - qa_factoid, qa_definition, qa_stock → ask_fact.
  - cooking_recipe → how_to. cooking_query → ask_fact.
  - takeaway_order, transport_ticket, transport_taxi → book_or_order. takeaway_query → check_status.
  - transport_query, transport_traffic → navigation_traffic.
  - recommendation_\* → find_or_recommend.
  - general_greet → greeting.
  - general_joke, general_quirky → small_talk. general_quirky is a noisy catch-all; **consider dropping it**.
  - music_likeness, music_dislikeness → drop (preference statements).
  - play_game → drop.
- **BANKING77.**
  - \*transfer\*, top_up\*, receiving_money, exchange_via_app → pay_or_transfer.
  - pending_\*, balance_not_updated\*, card_arrival, card_delivery_estimate, transfer_timing → check_status.
  - card_not_working, declined_\*, failed_transfer, top_up_failed, pin_blocked, card_swallowed, contactless_not_working, virtual_card_not_working, wrong_amount_of_cash_received, lost_or_stolen_card, lost_or_stolen_phone, Refund_not_showing_up, reverted/transaction_charged_twice, extra_charge_on_statement, card_payment_fee_charged, transfer_fee_charged → report_problem.
  - card_payment_not_recognised, cash_withdrawal_not_recognised, direct_debit_payment_not_recognised, compromised_card → fraud_or_security.
  - edit_personal_details, change_pin, passcode_forgotten, terminate_account, verify_my_identity, unable_to_verify_identity, why_verify_identity, verify_source_of_funds, activate_my_card, card_linking → account_management.
  - order_physical_card, get_physical_card, getting_spare_card, getting_virtual_card, get_disposable_virtual_card, card_about_to_expire → request_item_or_document.
  - request_refund, cancel_transfer → cancel. request_refund is ambiguous; drop it if the frozen map cannot place it.
  - age_limit, atm_support, card_acceptance, country_support, fiat_currency_support, supported_cards_and_currencies, visa_or_mastercard, disposable_card_limits, top_up_limits, exchange_rate, exchange_charge, cash_withdrawal_charge, top_up_by_\*_charge, apple_pay_or_google_pay, automatic_top_up, beneficiary_not_allowed, verify_top_up → ask_service_info.
- **SGD** (first user turn of each new `active_intent`).
  - Reserve\*, Book\*, Buy\*, GetRide, GetTrainTickets, ScheduleVisit, BookAppointment → book_or_order.
  - Find\*, Search\*, Lookup\*, GetCarsAvailable, GetEventDates, GetTimesForMovie → find_or_recommend.
  - CheckBalance → check_status.
  - TransferMoney, MakePayment, RequestPayment → pay_or_transfer.
  - AddEvent, GetEvents, GetAvailableTime, AddAlarm, GetAlarms → schedule_and_reminders.
  - PlaySong, PlayMedia, PlayMovie → play_media.
  - GetWeather → weather.
  - ShareLocation → communicate.
  - FindAttractions → find_or_recommend.
  - FindBus, FindTrains → find_or_recommend. These are not navigation; they are trip search.
- **MultiDoGO** (turn level; drop `contentonly` and every `<div>` multi-label).
  - bookflight, order\*intent, startorder, startserviceintent → book_or_order.
  - cancelserviceintent → cancel.
  - changeseatassignment, changeorder, upgradeserviceintent, transferserviceintent, updateaddress → modify. updateaddress would also fit account_management; **choose account_management**, as in ABCD.
  - checkbalance, checkclaimstatus, viewbillsintent, checkserverstatus → check_status.
  - transfermoney → pay_or_transfer.
  - reportlostcard, reportbrokenphone, reportbrokensoftware → report_problem.
  - disputecharge → fraud_or_security.
  - openaccount, closeaccount → account_management.
  - replacecard, getboardingpass, getproofofinsurance, orderchecks → request_item_or_document.
  - getseatinfo, getinformationintent, checkoffereligibility, getroutingnumber, getpromotions → ask_service_info.
  - openinggreeting, closinggreeting → greeting.
  - thankyou → thanks.
  - confirmation → affirm.
  - rejection → deny.
  - outofdomain → out_of_scope.
  - softwareupdate, expensereport → drop. Both are ambiguous between a request and a problem report.
- **ABCD** (first customer statement; label from subflow).
  - recover_\*, reset_2fa, manage_change_\*, manage_payment_method → account_management.
  - status_\* (under manage_account, order_issue, subscription), shipping status, refund_status, refund_update → check_status.
  - manage_upgrade, manage_downgrade, shipping manage → modify.
  - manage_create → book_or_order.
  - manage_cancel → cancel.
  - refund_initiate, return_\*, shipping missing, troubleshoot_site/\*, bad_price_\*, out_of_stock_\*, promo_code_\* → report_problem.
  - mistimed_billing_\*, manage_dispute_bill → fraud_or_security (disputed charge).
  - manage_pay_bill → pay_or_transfer.
  - manage_extension → modify.
  - single_item_query/\*, storewide_query/\*, shipping cost → ask_service_info.
- **MTOP (en).** Map its 117 top-level intents by domain prefix (alarm/timer/reminder/event → schedule_and_reminders; messaging/calling → communicate; music → play_media; weather → weather; news → read_news_or_messages; recipes → how_to/ask_fact). Its people domain (IN:GET_CONTACT, IN:GET_INFO_CONTACT…) → drop.
- **NLU++.** Multi-label; keep only examples whose intent set maps to exactly one class (e.g. {cancel, booking} → cancel; {not_working, …} → report_problem; {how_much, transfer_payment} → check_status); drop the rest.

### 1.4 Held-out breadth source

**Primary: MultiDoGO.** It is the only source whose authors, collection method (WOz
customer chats with trained agents) and domains (airline, fast food, finance,
insurance, media, software) differ from all the others. It covers about 15 of the 28 classes: book_or_order,
cancel, modify, check_status, pay_or_transfer, report_problem, fraud_or_security,
account_management, request_item_or_document, ask_service_info, greeting, thanks,
affirm, deny, and out_of_scope if used. Score balanced accuracy over **the classes present in the held-out
source**, with the model predicting over all 28. Pre-register that, and a minimum of
about 50 test items per class, capped per class so no class dominates.

**Secondary (reported, not gating): MTOP-en** for the assistant half (weather, schedule,
communicate, play_media, news). It is independent of MASSIVE by authorship, which MASSIVE
itself (since it shares text with HWU64) cannot be.

**Training pool:** CLINC150 + MASSIVE + BANKING77 + SGD + ABCD + NLU++ + SNIPS (both) +
MultiWOZ 2.2 (book/find only) + Braun SE (how_to/account). Bitext is optional and flagged
as synthetic.

### 1.5 Risks

- **Class balance.** Assistant intents (MASSIVE, CLINC) dominate the volume; customer-support intents come from ABCD, BANKING77 and SGD. Cap per class and per source.
- **Ambiguous boundaries.**
  - book_or_order vs find_or_recommend: SGD often opens with a search before it books.
  - report_problem vs fraud_or_security: disputed charges.
  - check_status vs ask_service_info.
  - modify vs account_management.
  Freeze every rule above before any number is seen.
- **Register mismatch.** MultiDoGO turns are mid-dialogue and often short ("yes please", "my card number is…"). Filter them with a pre-written rule: min 3 tokens, and never slot-only turns.
- **Leakage.**
  - HWU64 = MASSIVE text family: exclude HWU64.
  - BANKING77 vs NLU++ (same maker): run an exact and near-duplicate check (embedding cosine ≥ 0.95) before splitting.
  - MultiDoGO has no known overlap with the pool, but run the same check.

---

## 2. Finance: "which area of finance is this about"

### 2.1 Which question

A **finance-area** classifier is the most useful general question. It routes any
money question or complaint to the right area (and so to a team or a later
specialist). It can be fed from both consumer-question and complaint sources.
"Finance intent" (check balance, transfer…) is already covered by the intent
specialist above, so building it again would duplicate that work.

### 2.2 Candidates

| name | maker | URL | licence (where verified) | collection | personal data | size | labels | verdict |
|---|---|---|---|---|---|---|---|---|
| **CFPB Consumer Complaint Database, narratives archive** | US CFPB | consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/ ; files `files.consumerfinance.gov/f/documents/CCDB_Export_{1..21}_*.zip` (all Last-Modified 2026-09-14; e.g. Export 1 Dec 2011–Apr 2018 134.7 MB, Export 12 Jul–Aug 2025 84.2 MB, Export 21 Aug 2026 10.5 MB) | **No licence field on the archive page.** CFPB newsroom (cease-publication release): "The Bureau considers previously published narratives to be in the public domain for Freedom of Information Act (FOIA) purposes", proactively disclosed in the FOIA Reading Room. CFPB website policy: "Information created by the CFPB is in the public domain" (but narratives are written by consumers, not by the CFPB) | Consumer-written complaint narratives, published only with opt-in consent, scrubbed (XXXX) | scrubbed; ZIP3, state, company name kept | 21 zips covering Dec 2011–14 Aug 2026 | CSV header verified (Export 21): Date received, Product, Sub-product, Issue, Sub-issue, **Consumer complaint narrative**, Company public response, Company, State, ZIP code, Tags, Submitted via, Date sent to company, Company response to consumer, Timely response?, Complaint ID | **CAUTION → USE** (as in `free-datasets.md`). The archive now exists and is the source to snapshot: record SHA-256 per zip. See the date cut below |
| CFPB complaint database, live (no narratives) | US CFPB | consumerfinance.gov/data-research/consumer-complaints/ | as above | structured only since 2026-09-14 | — | — | product/issue only | **Not useful** (no text) |
| **Ask CFPB** | US CFPB | consumerfinance.gov/ask-cfpb/ | **Public domain**: "Information created by the CFPB is in the public domain and you may reproduce, publish, or otherwise use it without the Bureau's permission" (consumerfinance.gov/privacy/website-privacy-policy/) | Agency-written consumer questions + answers | none | count unverified (hundreds); no bulk download or API listed, so it must be crawled politely | 12 categories: Auto loans, Bank accounts, Credit cards, Credit reports and scores, Debt collection, Fraud and scams, Money transfers, Mortgages, Payday loans, Prepaid cards, Reverse mortgages, Student loans | **USE**. The question titles are short, consumer-phrased and labelled by the agency |
| **Personal Finance & Money Stack Exchange** | SE users | money.stackexchange.com | Content **CC BY-SA** (4.0 for recent posts; older posts 2.5/3.0), per SE. **But** the official dump now needs a login and agreement that it is "for my own use and for projects that do not include training a large language model" (meta.stackexchange.com/q/401324, reported by devclass 2024-07-30). Community copies of the 2025 dumps are on Academic Torrents / archive.org. The SE API (api.stackexchange.com) serves the same content; its API terms were not checked | Real people's questions, public posts | usernames (drop); questions may describe personal circumstances | **40,244 questions** (API `/info`, 2026-09-27) | Tags, top counts from the API: united-states 14,947, taxes 5,761, stocks 4,189, income-tax 3,724, investing 3,365, credit-card 1,759, mortgage 1,662, loans 1,159, 401k 1,146, banking 1,131, credit-score 890, ira 854, bonds 848, scams 724, retirement 722, insurance 520, debt 508, small-business 492, money-transfer 484, health-insurance 471, student-loan 423… | **CAUTION**. The content licence permits reuse, and we are not training an LLM. But the dump agreement is a contract, and the Reddit-style platform-terms question applies. Nick must decide. If accepted: API route, or a community dump with the hash recorded; posts ≤ 2022-11-01 only; attribute (BY-SA) |
| BANKING77 | PolyAI | see §1 | CC BY 4.0 (repo LICENSE) | see §1 | none | 13,083 | 77 banking intents | **USE**. Only 3–4 finance areas (bank_accounts, cards, payments, fraud) |
| CLINC150 banking + credit_cards (+ work: taxes, w2, rollover_401k, insurance, insurance_change, income, payday) | Clinc | see §1 | CC BY 3.0 | see §1 | none | 150 per intent | see §1 | **USE** |
| MultiDoGO finance + insurance | Amazon | see §1 | CDLA-Permissive-1.0 | see §1 | role-played | ~10k+ customer turns per domain | finance and insurance intents | **USE** (in finance, the domain name is the label: finance → by intent, insurance → insurance) |
| SGD Banks_1 / Payment_1 | Google | see §1 | CC BY-SA 4.0 | see §1 | none | — | CheckBalance, TransferMoney, Make/RequestPayment | **USE** (small) |
| IRS.gov pages (FAQs, Tax Topics) | US IRS | irs.gov | **Public domain**: "Content on this website that was created or maintained by federal employees in the course of their duties is not subject to copyright and may be freely copied" (irs.gov/about-irs/use-of-content-from-irsgov) | Agency-written | none | large | page section = tax sub-area | **USE** for the *tax* class (question-style FAQ titles preferred) |
| FTC consumer advice (consumer.ftc.gov) | US FTC | ftc.gov/policy-notices/website-policy | **Public domain** (17 USC 105), per the FTC website policy (via search snippet of the official page; the page itself was not fetched) | Agency-written | none | hundreds of articles | topics incl. scams, identity theft, credit, debt | **USE** for *fraud_scams* (and debt) |
| investor.gov (SEC), SSA.gov, healthcare.gov | US gov | — | presumed PD (17 USC 105); **unverified** per site | Agency-written | none | — | investing / retirement / insurance explainers | **USE if verified**. Needed for the investing, retirement and insurance classes if Money SE is refused |
| Wikipedia (finance article leads) | Wikipedia editors | via the existing `docs/wiki-data.md` pipeline (pre-2022-11-01 revisions) | CC BY-SA 4.0 (as in `wiki-data.md`) | Encyclopedic | none | thousands (category trees: Taxation, Insurance, Investment, Banking, Mortgage, Pensions, Fraud…) | category → area (Wikidata/category mapping) | **USE** as a register-diverse supplement, not alone. Its encyclopedic register is far from user questions |
| EDGAR-CORPUS | Loukas et al. 2021 | zenodo.org/records/5528490 (≈10.9 GB) | **CC BY 4.0** (Zenodo API metadata). Underlying 10-K text is filed by companies with the SEC | 10-K item sections, 1993–2020 | executives' names | ~11 GB | item section (Risk Factors, MD&A…) | **CAUTION / low value**. Corporate reporting, not consumer finance. The licence on company-authored 10-K text is the corpus maker's, not the companies'. Useful only for a "business_finance" class, and a small sample suffices |
| FinQA / ConvFinQA | Chen et al. (UCSB) | github.com/czyssrs/FinQA, /ConvFinQA | **MIT** (repo LICENSE) | Expert-written numeric questions over S&P 500 earnings reports | none | 8.3k / 3.9k | QA, no area labels | **USE licence-wise, not useful** (no area labels; all "corporate reporting") |
| TAT-QA | NExT++ (NUS) | github.com/NExTplusplus/TAT-QA | **Conflict**: GitHub LICENSE file = MIT; README §License = "Creative Commons (CC BY) Attribution 4.0 International". Both are permissive | Questions over financial report tables | none | 16.5k | QA | **USE licence-wise, not useful** (same reason) |
| FiQA 2018 | Maia et al. (WWW'18 challenge) | sites.google.com/view/fiqa | **Unverified**: no licence found; the BEIR/HF mirrors state one, but not at source | Crawled StackExchange (Investment), **Reddit, StockTwits** | usernames possible | 6,648 questions, 57k answers | QA; aspect sentiment | **NO**. No licence at source; Reddit and StockTwits text |
| Financial PhraseBank | Malo et al. | — | CC BY-NC-SA 3.0 | — | — | — | — | **NO** (NC; already in `free-datasets.md`) |
| twitter-financial-news-topic / finance-alpaca / Investopedia | various | — | — | Twitter / LLM-generated + Reddit / copyrighted | — | — | — | **NO** |

### 2.3 Proposed taxonomy: 15 areas (+ optional not_finance)

1. **bank_accounts**: checking/savings, deposits, account fees, cheques, routing, opening/closing
2. **cards**: credit, debit and prepaid cards: limits, APR, rewards, activation, lost/replacement
3. **payments_transfers**: P2P, wires, remittances, bill pay, top-ups, currency exchange, virtual currency as payment
4. **credit_reports_scores**: reports, scores, disputes of report entries, credit repair
5. **consumer_loans**: personal, payday, title and auto/vehicle loans and leases, lines of credit
6. **mortgages_home**: mortgages, refinancing, reverse mortgages, home equity, escrow, rental-property finance
7. **student_loans**
8. **debt_collection_relief**: collectors, debt management and settlement, bankruptcy
9. **investing**: stocks, funds, ETFs, bonds, options, brokerage, crypto as investment
10. **retirement**: 401(k), IRA, pensions, Social Security
11. **tax**: income, capital-gains and state tax; deductions; filing; IRS
12. **insurance**: health, auto, home, life, phone/device
13. **fraud_scams**: scams, identity theft, unauthorised transactions
14. **budgeting_saving**: budgeting, saving goals, emergency funds, financial literacy
15. **business_finance**: small-business finance, accounting, payroll, self-employment
16. (optional) **not_finance**: negatives from MASSIVE/CLINC non-finance intents. Recommended if the service will see arbitrary text; otherwise leave it out.

### 2.4 Mapping plan

- **CFPB Product (and Issue where needed).**
  - Checking or savings account / Bank account or service → bank_accounts.
  - Credit card / Credit card or prepaid card / Prepaid card → cards.
  - Money transfer, virtual currency, or money service / Money transfers / Virtual currency → payments_transfers.
  - Credit reporting\* → credit_reports_scores.
  - Payday loan, title loan, or personal loan (and its later variants) / Consumer Loan / Vehicle loan or lease → consumer_loans.
  - Mortgage → mortgages_home.
  - Student loan → student_loans.
  - Debt collection / Debt or credit management → debt_collection_relief.
  - Fraud: **override** to fraud_scams when Sub-issue or Issue contains "fraud or scam" or "identity theft", and the product is money transfer, bank account or card (pre-write the exact string list from the published field values, not from narratives).
  - Other financial service → drop.
- **Ask CFPB categories.**
  - Bank accounts → 1; Credit cards and Prepaid cards → 2; Money transfers → 3.
  - Credit reports and scores → 4; Auto loans and Payday loans → 5.
  - Mortgages and Reverse mortgages → 6; Student loans → 7.
  - Debt collection → 8; Fraud and scams → 13.
- **Money SE tags** (if accepted). Assign an area only when the question's tags hit **exactly one** area; otherwise drop.
  - taxes/income-tax/capital-gains-tax/tax-deduction/irs/state-income-tax → tax.
  - stocks/investing/etf/bonds/options/mutual-funds/index-fund/dividends/trading → investing.
  - 401k/ira/roth-ira/retirement/pension/social-security → retirement.
  - insurance/health-insurance/life-insurance/car-insurance → insurance.
  - credit-card → cards.
  - mortgage/real-estate/rental-property/refinance → mortgages_home.
  - loans/car-loan/personal-loan → consumer_loans.
  - student-loan → student_loans.
  - credit-score/credit-report → credit_reports_scores.
  - scams/fraud/identity-theft → fraud_scams.
  - debt/collections/bankruptcy → debt_collection_relief.
  - banking/bank-account/checking-account/savings-account/check → bank_accounts.
  - money-transfer/currency/foreign-exchange/paypal → payments_transfers.
  - budget/budgeting/savings/financial-literacy/emergency-fund → budgeting_saving.
  - small-business/self-employment/accounting/llc/limited-liability-company/payroll → business_finance.
  - Country tags (united-states, uk…) are ignored.
- **BANKING77.**
  - Account, identity and top-up-policy intents → bank_accounts.
  - Card intents → cards.
  - Transfer, top-up and exchange intents → payments_transfers.
  - \*_not_recognised, compromised_card → fraud_scams.
- **CLINC150.**
  - Banking domain → bank_accounts, except transfer/pay_bill → payments_transfers and report_fraud → fraud_scams.
  - credit_cards domain → cards, except credit_score/improve_credit_score → credit_reports_scores.
  - taxes, w2 → tax.
  - rollover_401k → retirement.
  - insurance, insurance_change → insurance.
  - exchange_rate → payments_transfers.
  - income, payday, direct_deposit → bank_accounts.
- **MultiDoGO.**
  - Insurance domain (task intents only) → insurance.
  - Finance domain: checkbalance, getroutingnumber, orderchecks, open/closeaccount, updateaddress → bank_accounts; reportlostcard, replacecard, checkoffereligibility → cards; transfermoney → payments_transfers; disputecharge → fraud_scams.
  - Dialogue acts → drop.
- **IRS / FTC / investor.gov / SSA / healthcare.gov pages.** The site is the label: tax, fraud_scams, investing, retirement, insurance. Use question-style titles and FAQ questions, not body text.
- **Wikipedia.** A fixed category → area map written in advance. The lead paragraph is the text.

### 2.5 Held-out breadth source

**Primary: the CFPB narratives archive**, restricted to complaints **received ≤ 2022-11-01**.
That is Export 1 (Dec 2011–Apr 2018), Export 2 (May 2018–Apr 2021), and the in-range part
of Export 3. This matches the project's pre-LLM text rule: complaints after late 2022 may
be LLM-drafted. Why this source:
- it is real consumer writing;
- it has a different register (complaint letters, long) from every training source;
- its labels are assigned by the complaint form, not by us.

It tests 9 areas: bank_accounts, cards, payments_transfers, credit_reports_scores,
consumer_loans, mortgages_home, student_loans, debt_collection_relief, fraud_scams.
Balanced accuracy is over those 9, with a fixed per-class sample (e.g. 300 each, seeded).
Truncate each narrative to a pre-registered length before embedding, since
nomic-embed has a context limit.

**Secondary (reported): Ask CFPB question titles.** Same 9 areas, agency-written. It is
short and clean, and a check that the model is not keying on complaint phrasing.

**Training pool:** Money SE (if accepted), BANKING77, CLINC150 finance/work intents,
MultiDoGO finance+insurance, SGD Banks/Payment, government FAQ pages (IRS, FTC, and
investor.gov/SSA/healthcare.gov once verified), and Wikipedia leads.

**Caveat:** this pool gives no training text from the CFPB archive. If gate 4 allows
both, an alternative is to train on the CFPB archive and hold out Money SE, which covers
all 15 areas. That makes a stronger breadth test, but it depends on the Money SE decision.

### 2.6 Risks

- **The areas the test cannot see.** investing, retirement, tax, insurance, budgeting and business_finance are absent from the CFPB held-out set. They are only tested if Money SE (or a held-out gov/Wikipedia slice) is used, so the breadth claim must name the 9 tested areas.
- **CFPB imbalance.** Credit reporting is the large majority of recent complaints. Sample per class; never use raw proportions.
- **Overlapping labels.**
  - cards vs bank_accounts vs payments_transfers inside BANKING77 (a card-centric neobank).
  - fraud_scams vs the product it happened on (CFPB).
  - mortgages_home vs debt_collection_relief (foreclosure).
  - credit_reports_scores vs debt_collection_relief (collections on a report).
  All of these need the written override rules above.
- **Register gap.** Training data is mostly short queries; the CFPB test items are long complaints. This is the point of a breadth test, but it may fail on length alone. Pre-register the truncation and consider a first-N-sentences rule.
- **Legal.**
  - CFPB narratives: consumer-authored, released as "public domain for FOIA purposes", with no formal licence (CAUTION → USE, as before).
  - Money SE: dump-agreement conflict (CAUTION).
  - Wikipedia, SGD and MTOP: BY-SA. Attribute, and keep derived data BY-SA.
- **Personal data.** CFPB narratives are scrubbed, but ZIP3/state/company columns must be dropped. Money SE usernames: drop.

---

## Sources checked (this pass)

- GitHub licence API / raw files. Commits:
  - xliuhw/NLU-Evaluation-Data `f6071b4`
  - google-research-datasets/dstc8-schema-guided-dialogue `e852981`
  - google-research-datasets/Taskmaster `d92cb6a`
  - budzianowski/multiwoz `fe0c8e6`
  - awslabs/multi-domain-goal-oriented-dialogues-dataset `baa3063`
  - asappresearch/abcd `6b8700c`
  - RasaHQ/STAR `3058975`
  - PolyAI-LDN/task-specific-datasets `57ec275`
  - sonos/nlu-benchmark `b86ac7f`
  - czyssrs/FinQA, czyssrs/ConvFinQA, NExTplusplus/TAT-QA (LICENSE = MIT; TAT-QA README says CC BY 4.0)
  - hellohaptik/HINT3 (LICENSE.md: ODbL + DbCL)
  - sebischair/NLU-Evaluation-Corpora (README: CC BY-SA 3.0)
  - AmFamMLTeam/ACID (no licence)
- Zip members read by range request: `mtop/LICENSE.txt` (CC BY-SA 4.0), `top-dataset-semantic-parsing/README` (CC-BY-SA), and the CFPB `CCDB_Export_21_August_2026.csv` header.
- consumerfinance.gov:
  - FOIA narratives archive page (21 zip links, no licence text);
  - newsroom "The CFPB to Cease Discretionary Publication…";
  - website privacy policy (public-domain statement);
  - Ask CFPB index (12 categories).
- irs.gov/about-irs/use-of-content-from-irsgov; FTC website policy (search snippet).
- api.stackexchange.com `/info` and `/tags` for site=money; meta.stackexchange.com/q/401324 via devclass.com (2024-07-30).
- zenodo.org/api/records/5528490 (EDGAR-CORPUS, cc-by-4.0).
- LDC licensing and ATIS catalog pages (via search); ACL Anthology D19-1460 (MultiDoGO abstract).
- Unverified: MTOP-en and STAR sizes, Snips-2017 authorship, Ask CFPB item count, the FTC page text (snippet only), investor.gov/SSA/healthcare.gov reuse statements, and the SE API terms.
