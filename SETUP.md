# AI Trading Research Assistant — Setup Guide
# Schwab + ThinkorSwim + Claude AI

## What This App Does
- Live quotes & options chains with full Greeks (Delta, Gamma, Theta, Vega, IV)
- Buy/Sell/Hold signals using RSI, SMA crossover, and ATR-based stop loss suggestions
- Paper trading with $30,000 starting capital — full P&L tracking
- Market & limit order entry with one-click confirmation
- AI research chat powered by Claude (ask anything about any ticker)
- Live portfolio updates via WebSocket

---

## Step 1 — Get Your API Keys

### Schwab Developer API
1. Go to https://developer.schwab.com
2. Create an account and log in
3. Click "Create App"
4. Set Callback URL to: https://127.0.0.1:8182
5. Copy your App Key (= SCHWAB_API_KEY) and Secret (= SCHWAB_SECRET)
6. Note: Your Schwab developer account must be linked to your brokerage account

### Anthropic API Key (Claude AI chat)
1. Go to https://console.anthropic.com
2. Create account → API Keys → Create Key
3. Copy the key (starts with sk-ant-...)

---

## Step 2 — Install Python & Dependencies

Make sure Python 3.10+ is installed, then:

```bash
# Navigate to the app folder
cd schwab_trader

# Install dependencies
pip install -r requirements.txt
```

---

## Step 3 — Configure Your .env File

```bash
# Copy the template
cp .env.example .env

# Edit .env and paste your keys
```

Your .env should look like:
```
SCHWAB_API_KEY=your_actual_key_here
SCHWAB_SECRET=your_actual_secret_here
SCHWAB_CALLBACK_URL=https://127.0.0.1:8182
TOKEN_PATH=schwab_token.json
ANTHROPIC_API_KEY=sk-ant-your_key_here
PAPER_TRADING=true
```

---

## Step 4 — Run the App

```bash
python main.py
```

On first run:
- A browser window will open for Schwab OAuth login
- Log in with your Schwab brokerage credentials
- You will be redirected to 127.0.0.1:8182 (this is expected)
- Copy the full redirect URL and paste it into the terminal
- Your token is saved to schwab_token.json for future runs

After authentication, the app opens at http://127.0.0.1:5000

---

## Step 5 — Using the App

1. Enter any ticker (e.g. AAPL, NVDA, SPY) in the search bar and click Analyze
2. Dashboard shows live price, signal, stop loss, RSI, SMA, chart
3. Options tab shows full chain with Greeks — select expiration dates
4. Trade tab — enter symbol, qty, order type → confirm paper trade
5. Portfolio tab — live P&L on all positions, updated every 15 seconds
6. AI Chat — ask Claude anything about the ticker, strategy, or your portfolio

---

## Switching to Live Trading

When you are confident in paper trading results:
1. Edit .env → set PAPER_TRADING=false
2. Restart the app
3. ⚠️ IMPORTANT: All order confirmations still require YOU to click confirm
   The app never places trades automatically without your explicit approval

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Schwab auth fails | Make sure callback URL matches exactly in developer portal |
| "Module not found" | Run pip install -r requirements.txt |
| No options data | Some tickers have no options (ETFs, small caps) |
| AI chat not working | Check ANTHROPIC_API_KEY in .env |
| Token expired | Delete schwab_token.json and restart — re-auth will happen |

---

## File Structure

```
schwab_trader/
├── main.py              ← Backend: Schwab API, paper trading, AI chat
├── templates/
│   └── index.html       ← Full dashboard UI
├── requirements.txt     ← Python dependencies
├── .env                 ← Your API keys (never share this)
├── .env.example         ← Template (safe to share)
└── schwab_token.json    ← Auto-created on first login (never share this)
```

---

## Important Disclaimers
- This tool is for research and paper trading purposes
- Nothing in this app constitutes financial advice
- Past signals do not guarantee future results
- Always do your own research before any real trade
- Never risk more than you can afford to lose
