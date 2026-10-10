# MyMarket: manual steps

Only the steps that genuinely need a human. Everything else is already run, see
the Status table in [README.md](README.md).

Rewritten for v0.3: project renamed to MyMarket, food staples only, consumer
electronics removed. The CPI steps from the old M2 are gone with it.

Order: M1, M3, M5, M6, M7, M8, M9. You can do M3 at any time. M5 can be done
while nothing else is happening. M6 needs M1 first.

UI labels on Kaggle, Colab and Claude Desktop change. If a button name differs,
follow what you see on screen.

Throughout, `python` means the project interpreter:
`.venv\Scripts\python.exe` on Windows, `.venv/bin/python` on Mac or Linux.

---

## M1. Get the Kaggle news dataset (15 min)

**Goal:** news files inside `data/raw/news/`. Nothing can be trained without this.

### Option A: browser
1. Go to kaggle.com. Sign up or log in.
2. Open `kaggle.com/datasets/azraimohamad/news-article-weekly-updated`.
3. Click **Download**. A zip downloads.
4. Unzip it. Put the files in `data/raw/news/`.

### Option B: command line
1. Kaggle, profile picture, **Settings**, **API**, **Create New Token**.
   A `kaggle.json` downloads.
2. Put it at `C:\Users\<you>\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json`
   then `chmod 600 ~/.kaggle/kaggle.json` (Mac or Linux).
3. Run:
```bash
uv pip install --python .venv/Scripts/python.exe kaggle
kaggle datasets download -d azraimohamad/news-article-weekly-updated -p data/raw/news --unzip
```

### Then
```bash
python scripts/02_prepare_news.py
```

**Check:** it prints a row count, a date range and the top domains, and writes
`data/processed/news_clean.parquet`. Put those numbers in the README.

**If it fails:**
- 401 or 403 from the CLI: the token is in the wrong folder, or you have not
  accepted the dataset terms. Open the dataset page in a browser once while
  logged in.
- "missing required columns": the dataset schema changed. The script prints what
  it found. The required columns are title, text, published_date, url.
- Out of memory: the script already reads CSVs in 50,000 row chunks. If it still
  fails, split the file.

---

## M3. Review the item mapping (20 min)

**Goal:** decide which PriceCatcher items represent each product.

A **provisional** `config/item_mapping.csv` already exists, picked by ranking
items on history length times average premise count. It is a defensible guess,
not a reviewed mapping. The price index you see now was built from it.

1. Open `data/processed/item_mapping_review.csv` in Excel or Sheets.
   130 candidate rows, with a `suggested` column and, importantly,
   `n_days`, `avg_premises` and `first_date`.
2. Keep 2 to 4 rows per product. Prefer a long `n_days` and a high
   `avg_premises`. An item surveyed at 3 premises is noise.
3. What to keep:
   - **rice**: plain white and imported rice packs. Not glutinous (PULUT), not
     basmathi, not baby food.
   - **egg**: chicken eggs grades A, B, C. Not quail (PUYUH), duck (ITIK),
     salted (MASIN) or free range (KAMPUNG).
   - **cooking_oil**: bottled or packed cooking oil. Note that the current
     provisional mapping included a corn oil (MINYAK JAGUNG); decide whether you
     want that. Beware item 918, the subsidised 1 kg packet, which is price
     controlled and nearly flat, so it will mute the index.
   - **chicken**: whole chicken standard or super, plus common cuts. Not turkey
     (AYAM BELANDA) or old hen (AYAM TUA).
4. Save the kept rows as `config/item_mapping.csv`, **CSV UTF-8**. It needs at
   least the columns `product` and `item_code`.
5. Rebuild:
```bash
python scripts/01_download_prices.py      # fast, resumes, re-filters to your codes
python scripts/03_build_price_index.py
```

**Check:** open the four PNGs in `data/processed/plots/`. They should look like
price curves, not random spikes. One or two step changes is normal and real:
rice genuinely jumped about 45 percent in September 2023.

**If it fails:**
- Flat line for a product: either you kept only price controlled items, or that
  product really was under a ceiling price. Eggs were flat until mid 2024 for
  exactly that reason.
- Spiky plot: drop the items with a low `avg_premises`.
- Empty index: your `item_code` values do not match PriceCatcher. Use the
  numbers straight from the review file.

---

## M5. Audit the relevance filter (45 min)

**Goal:** measure how often a matched article is really about price, and tune the
filter threshold on real labels instead of my guess.

Needs M1 done first.

1. Generate a sample:
```bash
python scripts/02_prepare_news.py --audit 120
```
2. Open `data/processed/audit_sample.csv`.
3. Fill the `relevant` column with `1` or `0` for every row.

Enter **1** when the article is about that product's price, supply, cost,
subsidy, import, shortage, or a policy affecting it.

Enter **0** when:
- the word is a brand or a name ("Ayam Brand", "Chicken Rice Shop")
- it is a recipe, a restaurant review or a lifestyle piece
- it is sport or entertainment and the word is a coincidence
- the sense is different: a flood story mentioning rice fields is 1 only if it
  says something about crop or supply

4. Save as CSV UTF-8, same name, same place.
5. Tune:
```bash
python scripts/02_prepare_news.py --tune-filter
```
6. It prints a precision and recall table and names the best threshold. Put that
   number in `config/products.yaml` under `filter.embed_threshold`.

**Check:** precision at least 80 percent.

**If precision is under 70 percent:** look at the rows you marked 0, find the
repeated noise phrases, add them to `exclude_phrases` for that product in
`config/products.yaml`, then re-run `--audit` with a fresh sample and label again.

Label all 120 in one sitting. Be strict and consistent.

---

## M6. Train the model (30 to 90 min)

**Goal:** four files in `models/`: `model.joblib`, `label_map.json`,
`thresholds.json`, `metrics.json`.

Needs M1. Do M3 and M5 first if you can, since both improve the labels.

```bash
python scripts/05_build_examples.py
```

Read what it prints before going further:
- label counts per product per split. Under 300 for a product means that
  product's result means little.
- the theta per product. If one says "at the floor", that product's price was
  nearly fixed during training and its labels are being decided by the floor
  rather than by the data. Expect weak results there and say so.
- the purge gap proof. If it is not ok, stop: the split leaks.

### Local CPU (simplest)
```bash
python scripts/06_train_local.py
python scripts/07_evaluate.py
```
Embedding is the slow part and is cached in `data/processed/embeddings.npy`.

### Colab with a GPU
1. drive.google.com, **New**, **Folder**, name it `mymarket`.
2. Upload `data/processed/examples.parquet` into it.
3. colab.research.google.com, **File**, **Upload notebook**,
   pick `notebooks/train_colab.ipynb`.
4. **Runtime**, **Change runtime type**, **T4 GPU**, Save. **Connect**.
5. Run the cells in order. Cell 2 checks the purge gap and will stop you if the
   split leaks.
6. Download the four files from Drive into `models/`.
7. Run `python scripts/07_evaluate.py` for the per product and rolling origin
   numbers.

### Read your result honestly
- Above majority **and** above momentum: good. Report it with the interval.
- About equal to the baselines: the normal outcome here. Say so, and show the
  keyword triggers and price context as the value instead.
- Below majority: not useful. Try `--horizon 28`, `--k 0.3`, or merge products.
- Above 0.9: suspicious. Check the purge gap for leakage.

**If it fails:**
- "examples.parquet not found": run `05_build_examples.py` first.
- "no article falls inside the price data window": your news predates 2022-01.
- Calibration says "skipped": the validation split is missing a class. The model
  still works, its probabilities are just less trustworthy, and `metrics.json`
  records it.
- Colab disconnects: rerun cell 1, then `emb = np.load(f'{P}/emb.npy')` instead
  of re-embedding.

---

## M7. Connect to Claude Desktop (15 min)

**Goal:** the tools usable in a chat.

### Step 1: check it with the Inspector first
```bash
npx @modelcontextprotocol/inspector uv run python -m mymarket.mcp_server
```
Open the printed URL, click **Connect**, open **Tools**, **List Tools**.
You should see 8: `list_domains`, `fetch_news`, `filter_relevant`,
`predict_impact`, `price_context`, `scan_alerts`, `backtest_report`,
`forward_validation`.

Try `list_domains`, then `price_context` with `product` = `chicken`.

### Step 2: add it to Claude Desktop
1. Get your absolute paths:
   - project: `(Get-Location).Path` (PowerShell) or `pwd` (Mac)
   - uv: `where.exe uv` (Windows) or `which uv` (Mac)
2. Claude Desktop, **Settings**, **Developer**, **Edit Config**.
3. Add this inside the existing `mcpServers` object. Do not create a second one.

Windows:
```json
{
  "mcpServers": {
    "mymarket": {
      "command": "C:\\Users\\UserPC\\Desktop\\MyMarket\\.venv\\Scripts\\python.exe",
      "args": ["-m", "mymarket.mcp_server"],
      "cwd": "C:\\Users\\UserPC\\Desktop\\MyMarket",
      "env": { "PYTHONPATH": "C:\\Users\\UserPC\\Desktop\\MyMarket\\src" }
    }
  }
}
```

Mac or Linux:
```json
{
  "mcpServers": {
    "mymarket": {
      "command": "/full/path/to/MyMarket/.venv/bin/python",
      "args": ["-m", "mymarket.mcp_server"],
      "cwd": "/full/path/to/MyMarket",
      "env": { "PYTHONPATH": "/full/path/to/MyMarket/src" }
    }
  }
}
```

4. Save. Quit Claude Desktop fully (Windows: close from the system tray.
   Mac: Cmd+Q). Reopen.
5. New chat, tools icon, `mymarket` should be listed.
6. Ask: "What Malaysian news in the last 2 days could affect food prices?"

**If it fails:**
- Not listed: JSON typo. Paste the file into a JSON validator and check commas.
- Disconnected or red: read the logs.
  `%APPDATA%\Claude\logs\` (Windows), `~/Library/Logs/Claude/` (Mac).
- Crash with JSON parse errors: something printed to stdout. The server is
  built to log only to stderr, and `tests/test_tools.py` checks this, so suspect
  a library you added.
- "No module named mymarket": the `PYTHONPATH` entry is wrong.

---

## M8. Schedule the daily job (15 min)

**Goal:** log live predictions so they can be scored 14 days later. This is what
makes `forward_validation` real rather than a promise.

Run it once by hand first:
```bash
python scripts/daily_refresh.py
```

### Windows, Task Scheduler
1. Open **Task Scheduler**, **Create Basic Task**.
2. Name: `MyMarket daily`.
3. Trigger: **Daily**, **14:00**.
4. Action: **Start a program**.
   - Program: `C:\Users\UserPC\Desktop\MyMarket\.venv\Scripts\python.exe`
   - Arguments: `scripts\daily_refresh.py`
   - Start in: `C:\Users\UserPC\Desktop\MyMarket`
5. Finish, then right click the task and **Run** to check it.

### Mac or Linux, cron
```bash
crontab -e
```
Add, with your own path:
```
0 14 * * * cd /path/to/MyMarket && .venv/bin/python scripts/daily_refresh.py >> data/daily.log 2>&1
```

Notes:
- 14:00 Malaysia time. PriceCatcher refreshes at 12:00, so 14:00 is safe.
- The machine must be awake and online.
- Predictions need 14 days to mature. Start now. By demo day you may have only a
  few scored rows. Say so rather than hiding it.

**Check:** the log shows "daily refresh finished" and the `predictions` table
gains rows. Without a trained model it will correctly log zero predictions and
still refresh prices.

---

## M9. Demo prep (30 min)

1. Run each demo question once and fix anything that errors.
2. Pick 2 alerts the model got right and 1 it got wrong from the backtest. Show
   the wrong one too. It is the most credible thing you can do.
3. Screenshot the metrics table: model against majority, momentum and keyword
   baselines, with the confidence interval.
4. One slide of limits:
   - Awareness alert, not a forecast.
   - The news signal is weak, by the nature of the problem.
   - PriceCatcher is a price surveillance survey. Inflation analysis uses DOSM CPI.
   - Rice in September 2023 is the clearest real event in the data, up about 45
     percent, and makes a good story.
5. Have screenshots of each tool result as a backup in case the Wi-Fi fails.

Demo script:
1. "What Malaysian news in the last 2 days could affect food prices?" (`scan_alerts`)
2. "Show the price trend for chicken." (`price_context`)
3. "How accurate is this model?" (`backtest_report`) and show the baselines.
4. "How did past live predictions do?" (`forward_validation`)
5. Close on the limits slide.

---

## Safety rules

1. Never commit `.env` or `kaggle.json`. Both are gitignored.
2. Never paste an API key into a chat. Put it in `.env` yourself. If a key has
   ever been pasted into a chat, rotate it.
3. Never enable a Scrapling stealth or anti bot fetcher.
4. Keep scraping to 1 request per second and cache everything.
5. Do not present predictions as financial advice.
6. Do not scrape Shopee or Lazada.
