# Gate 4 data: intent and finance, built 2026-09-27

Data for two gate-4 specialists (decision 29: specialists are generalists in their fields), pooled from clean
sources onto a taxonomy written before training, with a held-out source for the breadth test. Plan:
`docs/research-gate4/intent-financial.md`. Builder: `scripts/v1_gate4_intent_finance.py`
(`fetch`, `build`, `embed`, `ceilings`). Output: `data/cache/v1-gate4-intent-finance/` (`items.json`,
`emb.npz` with X, L, Z, ZL through the family antenna `service/families/v1/antenna.npz`). Machine-readable
provenance: `docs/gate4-data-manifest.json` (rows `intent/*` and `finance/*`) and `data/raw/clean/<source>/FETCH.json`.

Nothing was trained on the fly. Nothing was committed. The **sealed test** parts of the held-out sources were
counted and never scored.

**Order of work.** Sections 1–4 (taxonomies, maps, protocol) were written into the builder and this file before
any model was run on the data. The only things seen before that were label inventories, field values and a
handful of example utterances for labels whose meaning was unclear (`takeaway_query`, `transport_query`,
`qa_currency`, `music_query`, `music_settings`, `general_quirky`, `recommendation_locations`). Section 6
(ceilings) came last.

## 1. Sources

All licences were re-read at the source on 2026-09-27. User-Agent `bosco-research/0.1 (https://bosco.systems)`.

| source dir | used for | fetched from, version | licence (where verified today) | collection | personal data |
|---|---|---|---|---|---|
| `clinc150` (already fetched) | intent pool, finance pool | clinc/oos-eval `828f809` | CC BY 3.0 (repo LICENSE; `clean-data.md`) | crowd-written queries | none |
| `massive` (already fetched) | intent pool | MASSIVE 1.1 tarball, en-US | CC BY 4.0 (tarball LICENSE; `clean-data.md`) | SLURP text, crowd | `worker_id` never read |
| `snips_built_in_intents` (already fetched) | intent pool | sonos/nlu-benchmark `b86ac7f` | CC0 1.0 (repo LICENSE) | unknown authorship, 328 queries | none |
| `snips2017` | intent pool | sonos/nlu-benchmark `b86ac7f`, `2017-06-custom-intent-engines/*/train_*_full.json` + `validate_*.json` | CC0 1.0 (repo LICENSE, first line "CC0 1.0 Universal") | crowd queries | none; the `*_metrics.json` commercial-NLU outputs were not fetched |
| `polyai` (BANKING77, NLU++) | intent pool, finance pool (BANKING77 only) | PolyAI-LDN/task-specific-datasets `57ec275` | CC BY 4.0 (repo LICENSE is the CC BY 4.0 legal code; README "licensed under the license found in the LICENSE file"; GitHub API `CC-BY-4.0`) | online-banking queries; NLU++ written by dialogue experts | none |
| `sgd` | intent pool, finance pool (Banks/Payment) | google-research-datasets/dstc8-schema-guided-dialogue `e852981` (train/dev/test dialogues, 186 files) | **CC BY-SA 4.0** (README §License; LICENSE.txt) | simulator outlines paraphrased by paid crowd workers (2019) | none (fictional entities) |
| `abcd` | intent pool | asappresearch/abcd `6b8700c`, `data/abcd_v1.1.json.gz` | MIT (repo LICENSE) | crowd workers on both sides following a scenario | scenario name/e-mail/phone/address are generated; **never read**; customer text scrubbed |
| `multidogo` | **intent held-out**; finance pool (finance + insurance domains) | awslabs/multi-domain-goal-oriented-dialogues-dataset `baa3063`, turn-level paper splits, 6 domains | CDLA-Permissive-1.0 (LICENSE.txt first line; README §License) | Wizard-of-Oz, crowd customer + trained agent | role-played; scrubbed |
| `money_se` | finance pool | archive.org item `stackexchange_20240402` (Stack Exchange Data Dump 2024-04-02), `money.stackexchange.com.7z`; archive.org SHA-1 checked | **CC BY-SA** (dump `license.txt`: "cc-wiki licensed", links CC BY-SA 3.0, requires attribution; SE posts are 2.5/3.0/4.0 by date). **CAUTION accepted by Nick, decision 31**: posts before 2022-11-01 only; the newer download terms ("not training a large language model") go on the card | real people's public questions | Owner/Editor user fields **never read**; attribution by question id (URL `https://money.stackexchange.com/q/<unit>`); scrubbed |
| `cfpb_archive` | **finance held-out** | consumerfinance.gov FOIA reading room, CFPB Consumer Complaint Database Narratives Archive: Exports 1–3 (Dec 2011 – Oct 2022; Last-Modified 2026-09-14) | No licence field. CFPB newsroom (2026-08-14, "The CFPB to Cease Discretionary Publication of Complaint Narratives…", read today): "The Bureau considers previously published narratives to be in the public domain for Freedom of Information Act (FOIA) purposes". Website policy (read today): "Information created by the CFPB is in the public domain". CAUTION → USE as in `free-datasets.md` (narratives are consumer-written, opt-in, scrubbed by CFPB) | consumer complaint narratives, received ≤ 2022-10-31 (pre-LLM rule) | only Complaint ID, Date received, Product, Sub-product, Issue, Sub-issue, narrative kept; **state, ZIP, company, tags never kept**; `XXXX` masks stay; scrubbed |

Not used, with reasons: HWU64 (same text family as MASSIVE); MultiWOZ 2.2 (170 MB for find/book intents SGD
already covers); Taskmaster, STAR (no utterance-level intent); MTOP (secondary held-out in the plan; not built this
pass); Braun SE (≈500 items); Ask CFPB (secondary finance held-out; needs a polite crawl; not built this pass);
IRS/FTC/investor.gov pages and Wikipedia finance leads (planned supplements; not built this pass); Bitext
(synthetic); ACID, ATIS, FiQA, Financial PhraseBank, HINT3 (licence, NC or real-user issues; see the plan).

Zips and extraction: each CFPB zip's SHA-256 was recorded, only the columns above were written to
`*.narratives.csv.gz`, and the zip was deleted. Money SE `Posts.xml` was parsed (questions only, before
2022-11-01, not closed) into `questions_pre2022-11.jsonl.gz` and deleted; the 7z stays. Note: the 2024-04 dump's
`Posts.xml` is UTF-16 with a BOM although it declares UTF-8; it is read line by line.

## 2. Taxonomies

### 2.1 Intent (28 classes; design B)

As proposed in the plan, §1.2, unchanged: book_or_order, cancel, modify, check_status, pay_or_transfer,
report_problem, fraud_or_security, account_management, request_item_or_document, find_or_recommend,
ask_service_info, ask_fact, how_to, calculate_or_convert, ask_time_date, weather, navigation_traffic,
schedule_and_reminders, lists_and_notes, play_media, device_control, communicate, read_news_or_messages,
greeting, thanks, affirm, deny, small_talk.

**out_of_scope is not a class.** CLINC `oos` holds queries outside CLINC's 150 intents, many of which fall inside
this taxonomy (a query about a doctor is find_or_recommend here); MultiDoGO `outofdomain` is out of *its* six
domains, not ours. Both are dropped.

### 2.2 Finance (15 areas; design B)

As proposed in the plan, §2.3: bank_accounts, cards, payments_transfers, credit_reports_scores, consumer_loans,
mortgages_home, student_loans, debt_collection_relief, investing, retirement, tax, insurance, fraud_scams,
budgeting_saving, business_finance. **not_finance is left out** (the plan's option): the caller routes to this
specialist; an out-of-field detector is a separate question for Nick.

Service option strings are the class names with `_` → space.

## 3. Source → taxonomy maps (frozen; the code's dicts are the canonical copy)

A label not listed is **dropped**. Dropped counts per label are in `items.json` → `build_report.*.labels_dropped`.

### 3.1 Intent

Changes from the plan's draft, with reasons:
- the draft's double listings are resolved: `accept_reservations` → ask_service_info; `calendar_update`,
  `reminder_update` → schedule_and_reminders; CLINC `exchange_rate` and MASSIVE `qa_currency` both →
  calculate_or_convert (same query type in both sources; the draft split them);
- CLINC `credit_limit_change`, `insurance_change` → modify (the draft sent credit_limit_change to device_control,
  a slip: it is a request to change a credit limit);
- MASSIVE `general_quirky` dropped (the draft's suggestion: a noisy catch-all) and `transport_query` dropped (mixes
  fares, directions and ticket requests);
- ABCD `status_mystery_fee` dropped (a status question about an unrecognised charge: check_status or
  fraud_or_security);
- BANKING77 `request_refund` dropped (as the draft allowed);
- MultiDoGO `providereceipt` dropped (the customer supplies a receipt; not a goal);
- MultiDoGO `updateaddress` → account_management (the draft's own choice).

**CLINC150** — book_or_order: book_flight, book_hotel, car_rental, restaurant_reservation, uber, order ·
cancel: cancel_reservation · modify: credit_limit_change, insurance_change · check_status: flight_status,
order_status, application_status, balance, bill_balance, bill_due, transactions, spending_history, pto_balance,
pto_request_status, rewards_balance, confirm_reservation, credit_score · pay_or_transfer: pay_bill, transfer ·
report_problem: card_declined, damaged_card, lost_luggage, report_lost_card · fraud_or_security: report_fraud,
freeze_account, account_blocked · account_management: pin_change, user_name, change_user_name, direct_deposit ·
request_item_or_document: new_card, replacement_card_duration, order_checks, w2 · find_or_recommend:
travel_suggestion, meal_suggestion, restaurant_suggestion · ask_service_info: apr, interest_rate,
international_fees, credit_limit, carry_on, expiration_date, routing, travel_alert, international_visa,
plug_type, what_can_i_ask_you, redeem_rewards, accept_reservations, how_busy, restaurant_reviews, pto_used,
payday, income, insurance, taxes, rollover_401k, min_payment · ask_fact: definition, fun_fact, calories,
nutrition_info, meaning_of_life, vaccines, gas_type, mpg, food_last, ingredients_list · how_to: recipe,
cook_time, ingredient_substitution, oil_change_how, jump_start, tire_change, improve_credit_score ·
calculate_or_convert: calculator, measurement_conversion, spelling, translate, roll_dice, flip_coin,
exchange_rate · ask_time_date: time, date, next_holiday, timezone · weather · navigation_traffic: directions,
distance, traffic, current_location, gas, find_phone · schedule_and_reminders: alarm, timer, reminder,
reminder_update, calendar, calendar_update, meeting_schedule, schedule_meeting, schedule_maintenance,
pto_request · lists_and_notes: shopping_list(_update), todo_list(_update) · play_media: play_music, next_song,
update_playlist, what_song · device_control: smart_home, change_volume, change_speed, change_accent,
change_language, change_ai_name, whisper_mode, reset_settings, sync_device, repeat · communicate: make_call,
text, share_location · greeting: greeting, goodbye · thanks: thank_you · affirm: yes · deny: no · small_talk:
are_you_a_bot, how_old_are_you, tell_joke, what_are_your_hobbies, what_is_your_name, where_are_you_from,
who_do_you_work_for, who_made_you, do_you_have_pets. **Dropped:** maybe, cancel, last_maintenance,
tire_pressure, oil_change_when, travel_notification, oos.

**MASSIVE** — schedule_and_reminders: alarm_\*, calendar_\* · lists_and_notes: lists_\* · play_media:
play_music, play_radio, play_podcasts, play_audiobook, music_query, music_settings · device_control: iot_\*,
audio_volume_\* · communicate: email_sendemail, email_addcontact, social_post · read_news_or_messages:
email_query, email_querycontact, social_query, news_query · weather: weather_query · ask_time_date:
datetime_query · calculate_or_convert: datetime_convert, qa_maths, qa_currency · ask_fact: qa_factoid,
qa_definition, qa_stock, cooking_query · how_to: cooking_recipe · book_or_order: takeaway_order,
transport_ticket, transport_taxi · check_status: takeaway_query · navigation_traffic: transport_traffic ·
find_or_recommend: recommendation_\* · greeting: general_greet · small_talk: general_joke. **Dropped:**
general_quirky, music_likeness, music_dislikeness, play_game, transport_query.

**SNIPS built-in** — communicate: ShareCurrentLocation, ShareETA · find_or_recommend: SearchPlace ·
book_or_order: BookRestaurant, RequestRide · navigation_traffic: GetDirections, GetTrafficInformation ·
weather: GetWeather. **Dropped:** ComparePlaces, GetPlaceDetails.

**SNIPS 2017** — play_media: AddToPlaylist, PlayMusic · book_or_order: BookRestaurant · weather: GetWeather ·
find_or_recommend: SearchCreativeWork, SearchScreeningEvent. **Dropped:** RateBook.

**BANKING77** — pay_or_transfer: transfer_into_account, receiving_money, exchange_via_app, topping_up_by_card,
top_up_by_cash_or_cheque, automatic_top_up · check_status: pending_\* (4), balance_not_updated_\* (2),
card_arrival, card_delivery_estimate, transfer_timing, transfer_not_received_by_recipient · report_problem:
card_not_working, declined_\* (3), failed_transfer, top_up_failed, top_up_reverted, pin_blocked, card_swallowed,
contactless_not_working, virtual_card_not_working, wrong_amount_of_cash_received, lost_or_stolen_card,
lost_or_stolen_phone, Refund_not_showing_up, reverted_card_payment?, transaction_charged_twice,
extra_charge_on_statement, card_payment_fee_charged, transfer_fee_charged, card_payment_wrong_exchange_rate,
wrong_exchange_rate_for_cash_withdrawal · fraud_or_security: \*_not_recognised (3), compromised_card ·
account_management: edit_personal_details, change_pin, passcode_forgotten, terminate_account,
verify_my_identity, unable_to_verify_identity, why_verify_identity, verify_source_of_funds, activate_my_card,
card_linking · request_item_or_document: order_physical_card, get_physical_card, getting_spare_card,
getting_virtual_card, get_disposable_virtual_card, card_about_to_expire · cancel: cancel_transfer ·
ask_service_info: age_limit, atm_support, card_acceptance, country_support, fiat_currency_support,
supported_cards_and_currencies, visa_or_mastercard, disposable_card_limits, top_up_limits, exchange_rate,
exchange_charge, cash_withdrawal_charge, top_up_by_bank_transfer_charge, top_up_by_card_charge,
apple_pay_or_google_pay, beneficiary_not_allowed, verify_top_up. **Dropped:** request_refund.

**NLU++** (multi-label; only these patterns kept, all else dropped): {affirm}(+acknowledge) → affirm;
{deny}(+acknowledge) → deny; {greet} or {end_call} → greeting; {thank} → thanks; booking/appointment with the
single verb cancel_close_leave_freeze → cancel, change → modify, make_open_apply_setup_get_activate →
book_or_order; wrong_notworking_notshowing or lost_stolen with none of those verbs → report_problem; balance with
only request_info/how_much/account-type modifiers → check_status.

**SGD** (a user turn whose frame's `active_intent` differs from that service's previous one; turns opening two
intents at once dropped) — book_or_order: Reserve\*, BookHouse, BookAppointment, GetRide, GetTrainTickets,
Buy\*Ticket(s), ScheduleVisit, RentMovie · find_or_recommend: Find\*, Search\*, GetCarsAvailable, GetEventDates,
GetTimesForMovie, LookupMusic, LookupSong · check_status: CheckBalance · pay_or_transfer: TransferMoney,
MakePayment, RequestPayment · schedule_and_reminders: AddEvent, GetEvents, GetAvailableTime, AddAlarm, GetAlarms ·
play_media: PlaySong, PlayMedia, PlayMovie · weather: GetWeather · communicate: ShareLocation.

**ABCD** (the first customer turn of ≥ 5 words; label = the dialogue's subflow) — account_management:
recover_username, recover_password, reset_2fa, manage_change_address/name/phone, manage_payment_method ·
check_status: status_service_added/removed, status_shipping_question, status_credit_missing,
status_delivery_time, status_payment_method, status_quantity, refund_update, refund_status, shipping status,
status_active, status_due_amount, status_due_date, status_delivery_date, status_questions · modify: manage_upgrade, manage_downgrade, shipping manage,
manage_extension · book_or_order: manage_create · cancel: manage_cancel · report_problem: refund_initiate,
return_\*, shipping missing, troubleshoot_site (credit_card, shopping_cart, search_results, slow_speed),
bad_price_\*, out_of_stock_\*, promo_code_\* · fraud_or_security: mistimed_billing_\*, manage_dispute_bill ·
pay_or_transfer: manage_pay_bill · ask_service_info: every single_item_query and storewide_query subflow
(the data's subflows are finer than `ontology.json`: boots_how_1 … timing_4), shipping cost. (The first build
missed these finer names and `status_delivery_date`, `status_questions`; fixed from the dropped-label report
before any embedding finished or model ran.) **Dropped:** status_mystery_fee.

**MultiDoGO (held-out)** (turn level, ≥ 3 words; `<div>` multi-labels dropped) — book_or_order: bookflight,
order\*intent (7), startserviceintent, startorder · cancel: cancelserviceintent, stoporder · modify:
changeseatassignment, changeorder, upgradeserviceintent, transferserviceintent · check_status: checkbalance,
checkclaimstatus, viewbillsintent, checkserverstatus, viewdatausageintent · pay_or_transfer: transfermoney ·
report_problem: reportlostcard, reportbrokenphone, reportbrokensoftware · fraud_or_security: disputecharge ·
account_management: updateaddress, openaccount, closeaccount, updateaccountinfo · request_item_or_document:
replacecard, getboardingpass, getproofofinsurance, orderchecks · ask_service_info: getseatinfo,
getinformationintent, checkoffereligibility, getroutingnumber, getpromotions, getchannelpackageintent ·
greeting: openinggreeting, closinggreeting · thanks: thankyou · affirm: confirmation · deny: rejection.
**Dropped:** contentonly (slot-only turns), outofdomain, softwareupdate, expensereport, providereceipt.

### 3.2 Finance

Changes from the plan's draft, with reasons:
- Money SE: the draft's tag lists were extended with tags that exist in the dump and name one area without doubt
  (e.g. form-1099, withholding → tax; brokerage, call-options → investing; roth-401k, rrsp → retirement;
  debit-card → cards; home-loan, first-time-home-buyer → mortgages_home; auto-loan → consumer_loans; wire-transfer
  → payments_transfers; bookkeeping, s-corporation → business_finance). The full lists are `SE_FIN` in the code.
  Closed questions are dropped (often off-topic).
- CLINC: min_payment → cards (it is a credit-card minimum payment; CLINC files it under banking);
  bill_balance, bill_due, income, payday dropped (utility bills and pay-day questions; no clean area).
- BANKING77: the draft's "top-up-policy → bank_accounts" is replaced by one rule: every top-up, transfer and
  exchange label → payments_transfers (below).

**CFPB (held-out)**, Product → area: Checking or savings account, Bank account or service → bank_accounts ·
Credit card or prepaid card, Credit card, Prepaid card → cards · Money transfer, virtual currency, or money
service; Money transfers; Virtual currency → payments_transfers · Credit reporting, credit repair services, or
other personal consumer reports; Credit reporting → credit_reports_scores · Payday loan, title loan, or personal
loan; Consumer Loan; Vehicle loan or lease; Payday loan → consumer_loans · Mortgage → mortgages_home · Student loan
→ student_loans · Debt collection → debt_collection_relief. **Dropped:** Other financial service.
**Fraud override** → fraud_scams, exact (Product, Issue, Sub-issue) values from the published fields:
(Money transfer, virtual currency, or money service | Fraud or scam | —), (Money transfers | Fraud or scam | —),
(Prepaid card | Fraud or scam | —), (Credit card or prepaid card | Getting a credit card | Card opened as result
of identity theft or fraud), (Checking or savings account | Opening an account | Account opened as a result of
fraud), (Credit card | Identity theft / Fraud / Embezzlement | —). Not overridden, on purpose: "Debt was result of
identity theft" (a debt-collection complaint), "Problem with fraud alerts or security freezes" and "Credit
monitoring or identity theft protection services" (credit-report products).

**Money SE** (a question is kept only if its tags hit **exactly one** area; country and other tags ignored):
tax: taxes, income-tax, capital-gains-tax, tax-deduction, irs, state-income-tax, gift-tax, withholding,
form-1099, form-w-2, form-w-4, income-tax-refund, tax-credit, property-taxes, sales-tax, vat, hst, payroll-taxes,
capital-gain, capital-loss, wash-sale · investing: stocks, investing, etf, bonds, options, mutual-funds,
index-fund, dividends, trading, stock-markets, stock-analysis, shares, investment-strategies,
starting-out-investing, stock-valuation, brokerage, broker, futures, shorting-securities, call-options,
put-options, portfolio, day-trading, cryptocurrency, bitcoin, technical-analysis, option-strategies,
derivatives, commodities, margin, ipo, online-brokerage, online-trading, limit-order, asset-allocation,
value-investing, diversification, government-bonds, fixed-income · retirement: 401k, ira, roth-ira, retirement,
pension, social-security, retirement-plan, roth-401k, rollover, rrsp, annuity, roth-conversion · insurance:
insurance, health-insurance, life-insurance, car-insurance · cards: credit-card, debit-card · mortgages_home:
mortgage, real-estate, rental-property, refinance, first-time-home-buyer, home-loan, home-ownership,
down-payment, mortgage-qualification · consumer_loans: loans, car-loan, auto-loan, personal-loan ·
student_loans: student-loan · credit_reports_scores: credit-score, credit-report, credit-history · fraud_scams:
scams, fraud, identity-theft · debt_collection_relief: debt, debt-collection, collections, bankruptcy ·
bank_accounts: banking, bank-account, checking-account, savings-account, check, online-banking, deposits ·
payments_transfers: money-transfer, currency, foreign-exchange, paypal, international-transfer, wire-transfer,
online-payment · budgeting_saving: budget, budgeting, savings, financial-literacy, emergency-fund, expenses ·
business_finance: small-business, self-employment, accounting, llc, limited-liability-company, payroll,
s-corporation, bookkeeping, double-entry, contractor, start-up. Text = title + ". " + body (HTML stripped).

**BANKING77** (one rule over the 77 labels): the four fraud_or_security labels above → fraud_scams; any label
containing top_up / topping_up / transfer / exchange, plus receiving_money, fiat_currency_support,
beneficiary_not_allowed, verify_top_up → payments_transfers; else any label containing card, plus pin_blocked,
change_pin, contactless_not_working, visa_or_mastercard, apple_pay_or_google_pay, activate_my_card → cards;
everything else (identity, account, cash withdrawal, refunds, statements, ATM, country/age) → bank_accounts.

**CLINC150** (finance intents only; all others are not read into this field): bank_accounts: transactions,
balance, freeze_account, account_blocked, interest_rate, routing, order_checks, spending_history, direct_deposit ·
payments_transfers: transfer, pay_bill, exchange_rate · fraud_scams: report_fraud · cards: report_lost_card,
credit_limit, rewards_balance, new_card, application_status, card_declined, international_fees, apr,
redeem_rewards, credit_limit_change, damaged_card, replacement_card_duration, expiration_date, min_payment ·
credit_reports_scores: credit_score, improve_credit_score · tax: taxes, w2 · retirement: rollover_401k ·
insurance: insurance, insurance_change.

**MultiDoGO** (finance and insurance domains only, ≥ 3 words): bank_accounts: checkbalance, getroutingnumber,
orderchecks, openaccount, closeaccount, updateaddress · cards: reportlostcard, replacecard, checkoffereligibility
· payments_transfers: transfermoney · fraud_scams: disputecharge · insurance: getproofofinsurance,
checkclaimstatus, reportbrokenphone. (MultiDoGO is the *intent* held-out; using it in the *finance* pool does not
touch the intent breadth test, which is a different specialist.)

**SGD**: CheckBalance → bank_accounts; TransferMoney, MakePayment, RequestPayment → payments_transfers.

## 4. Protocol

- **Personal data** (all text, on read): e-mail addresses → `<email>`; runs of ≥ 7 digits with separators
  (phones, card and account numbers) → `<number>` (a four-digit year range is left alone); any other
  whitespace token with ≥ 7 digits (IBANs, member ids) → `<number>` unless it is a URL or path; CFPB `XXXX` masks stay.
  Username, annotator, worker, owner and editor fields are never read.
- **Pool dedupe:** exact duplicates (lower-cased text) kept once; a text that carries two classes is dropped entirely.
- **Pool splits:** by natural unit (SGD dialogue, ABCD conversation, MultiDoGO conversation, Money SE question,
  query text for query lists), hashed with seed 20260927: 15% `val`, 85% `train`.
- **Caps** (seeded): intent train ≤ 400 per class per source and ≤ 800 per class; intent val ≤ 60 / 150.
  Finance train ≤ 1,000 per class per source and ≤ 1,300 per class; finance val ≤ 150 / 250. Both fields
  stay under ~20k train.
- **Held-out:** one copy of each exact text; texts with two classes dropped; ≤ 300 per class (seeded); split by
  unit (MultiDoGO conversation, CFPB complaint) 20% `dev` / 80% `test`. **`test` is sealed**: counted here, never
  scored. A class with < 50 sealed-test items is reported as not testable.
- **Near duplicates** (nomic embeddings, cosine > 0.95, within field): pool items (train and val) that
  near-duplicate any held-out item (dev or test) are dropped from the pool; then val items that near-duplicate
  a train item are dropped. Exact pool/held-out duplicates are dropped at build time.
- **Length:** every text is cut to its first 200 words at embedding (the pinned pipeline's cut, as in
  `scripts/v1_data.py`); for CFPB narratives this is the pre-registered truncation.
- **Held-out balanced accuracy** is over the classes present in the held-out source, with the model predicting
  over the whole taxonomy.

## 5. What was built (after embedding and near-duplicate drops)

`data/cache/v1-gate4-intent-finance/items.json` (schema as gates 2/3, plus `source`, `unit`, `src_label` per
item; `build_report` and `near_dup` blocks) and `emb.npz` (X 768-d, L labels, Z/ZL 46-d via the family antenna).

| field | options | design | train | val | held-out dev | held-out test (sealed) |
|---|---|---|---|---|---|---|
| intent | 28 | B | 18,167 | 2,120 | 799 (MultiDoGO) | 3,092 |
| finance | 15 | B | 12,818 | 1,702 | 527 (CFPB) | 2,173 |

Drops: intent pool exact duplicates 5,691, texts with two classes 207, exact duplicates of held-out 4, **pool near
duplicates of held-out 257** (228 train, 29 val; MultiDoGO shares short acts such as "thank you so much" with
CLINC/NLU++), val near-duplicates of train 989. Finance: exact duplicates 2,397, two-class texts 3, near
duplicates of held-out **0**, val near-duplicates of train 535.

Train by source — intent: CLINC150 7,097, MASSIVE 4,256, BANKING77 2,000, ABCD 1,841, SGD 1,609, SNIPS 2017 813,
NLU++ 417, SNIPS built-in 134. Finance: Money SE 7,208, CLINC150 1,840, BANKING77 1,838, MultiDoGO 1,458, SGD 474.
Per-class counts are printed by `embed` and stored in `items.json`.

**Thin classes.** Intent train: thanks 113, affirm 150, deny 167, greeting 267 (CLINC + NLU++ only). Finance
train: **student_loans 115, debt_collection_relief 156** (Money SE only, after the one-area rule), consumer_loans
409, budgeting_saving 374, business_finance 432 (the last two from Money SE alone).

**Held-out coverage.** Intent (MultiDoGO): 14 of 28 classes present; **13 testable** (≥ 50 sealed-test items):
book_or_order, modify, check_status, pay_or_transfer, report_problem, fraud_or_security, account_management,
request_item_or_document, ask_service_info, greeting, thanks, affirm, deny. `cancel` has 13 sealed-test items: not
testable. Finance (CFPB ≤ 2022-10-31): **9 of 15 testable**: bank_accounts, cards, payments_transfers,
credit_reports_scores, consumer_loans, mortgages_home, student_loans, debt_collection_relief, fraud_scams.
investing, retirement, tax, insurance, budgeting_saving, business_finance have no held-out test.

## 6. Ceilings (report-only; pool val and held-out DEV only)

Logistic regression (sklearn, lbfgs, C ∈ {0.1, 1, 10} chosen on pool val); balanced accuracy. Held-out numbers
are over the classes present in the held-out dev (intent: 14, finance: 9), predicting over the whole taxonomy.
Nearest prototype = cosine to the class mean of train X. Full numbers, per-class recall and top confusions:
`runs/gate4-ceilings/intent-finance.json`. The sealed test was not scored.

| field | representation | C | pool val | held-out dev | held-out dev, testable classes |
|---|---|---|---|---|---|
| intent | X 768 (LR) | 10 | 0.876 | 0.581 | 0.575 |
| intent | Z 46 (LR) | 10 | 0.766 | 0.532 | 0.522 |
| intent | X 768 prototype | — | 0.751 | 0.575 | — |
| finance | X 768 (LR) | 10 | 0.839 | 0.502 | 0.502 |
| finance | Z 46 (LR) | 10 | 0.743 | 0.396 | 0.396 |
| finance | X 768 prototype | — | 0.757 | 0.293 | — |

C = 10 is the top of the grid in all four LR fits; a larger C might add a little on pool val. The grid was fixed
in advance and was not widened.

Held-out dev recall, LR on X (dev has 50–76 items per class, so each recall is ± ~0.06):
- intent: pay_or_transfer 0.94, account_management 0.84, report_problem 0.80, thanks 0.77, check_status 0.72,
  fraud_or_security 0.71, greeting 0.70, modify 0.49, affirm 0.49, ask_service_info 0.35, book_or_order 0.30,
  deny 0.19, request_item_or_document 0.18 (cancel 2/3 on 3 items). Top confusions: deny → thanks/greeting,
  modify → book_or_order, request_item_or_document → schedule_and_reminders/check_status.
- finance: fraud_scams 0.75, mortgages_home 0.64, cards 0.52, bank_accounts 0.50, consumer_loans 0.48,
  student_loans 0.47, payments_transfers 0.42, credit_reports_scores 0.38, debt_collection_relief 0.36. Top
  confusions: every area → fraud_scams (complaint letters read as fraud reports), payments → bank_accounts.

**What this says for the pre-registration** (not a decision; Nick's): even an unconstrained linear read of the
encoder reaches 0.88 / 0.84 on the pooled sources but only 0.58 / 0.50 on the held-out sources, far under the 0.80
bar. The fly, reading through the 46-d antenna, has a lower ceiling still (0.52 / 0.40 held-out). As built,
neither field can pass a breadth test at 0.80 on these held-out sources. The gap is register (MultiDoGO's
mid-dialogue turns; CFPB's long complaint letters) and label granularity (MultiDoGO "rejection" turns such as "no
that's all, thanks"), not only data volume.

## 7. Open, for Nick

1. **The breadth bar is out of reach as built** (§6). Options: (a) pre-register the held-out bar as a fraction of
   the held-out linear ceiling on the same data (decision 23's disputed-labels rule, applied to breadth);
   (b) finance: train on the CFPB archive (it is the only large source for student loans, debt collection,
   credit reports and consumer loans) and hold out Money SE instead, which tests all 15 areas (the plan's
   caveat, §2.5); (c) intent: add a dialogue source to the pool (e.g. MultiWOZ acts, Taskmaster) so turn-level
   acts are not learned from CLINC's `yes`/`no` alone.
2. **Thin finance classes** (student_loans 115, debt_collection_relief 156; budgeting_saving and business_finance
   from Money SE alone). Relaxing the one-area rule or adding government FAQ pages / Wikipedia leads would help.
3. **Secondary held-outs not built:** MTOP-en (intent, assistant half) and Ask CFPB (finance). Both are licence-clean
   (plan §1.4, §2.5); Ask CFPB needs a polite crawl.
4. **Share-alike:** SGD (both fields) and Money SE (finance) are CC BY-SA; under decision 28 the intent and finance
   specialists ship CC BY-SA 4.0 with an attribution manifest (SGD repo; Money SE question URLs from `unit`).
   Money SE authors' names were not extracted; if per-author attribution is wanted, re-read `OwnerDisplayName`
   from the 7z into the attribution manifest only.
5. **Manifest file name:** rows are in `docs/gate4-data-manifest-intent-finance.json` (the two sibling gate-4
   builds write per-field manifests too); merge into one `docs/gate4-data-manifest.json` if wanted.
