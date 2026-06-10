#!/usr/bin/env python3
"""
Morning Email — Daily Best Bets Summary

Sends an email at 7 AM with today's top predictions and value bets.

Requirements:
- Set EMAIL_SENDER, EMAIL_RECIPIENT, EMAIL_PASSWORD in .env
- For Gmail: use app-specific password with SMTP
- For other providers: configure SMTP_HOST and SMTP_PORT

Usage:
  python scripts/send_morning_email.py
  python scripts/send_morning_email.py --date 2026-06-09
  python scripts/send_morning_email.py --test  # dry run without sending

Environment variables:
  EMAIL_SENDER      — sender email address
  EMAIL_RECIPIENT   — recipient email address
  EMAIL_PASSWORD    — SMTP password (app-specific for Gmail)
  SMTP_HOST         — SMTP server (default: smtp.gmail.com)
  SMTP_PORT         — SMTP port (default: 587)
"""

import argparse
import os
import smtplib
import sys
from datetime import date, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

load_dotenv(ROOT / ".env")


def load_predictions(target_date: date) -> pd.DataFrame | None:
    """Load predictions CSV for target date."""
    pred_file = DATA_DIR / f"predictions_{target_date}.csv"
    
    if not pred_file.exists():
        return None
    
    df = pd.read_csv(pred_file)
    return df


def format_email_body(predictions: pd.DataFrame, target_date: date) -> str:
    """Generate HTML email body with top predictions."""
    
    # Filter for value bets (edge >= 5%)
    if 'edge' in predictions.columns:
        value_bets = predictions[predictions['edge'] >= 5.0].copy()
    else:
        # Calculate edge if not present
        predictions['model_odds'] = predictions['win_probability'].apply(
            lambda p: 1.0 / p if p > 0 else 999.0
        )
        if 'market_odds' in predictions.columns:
            predictions['edge'] = (
                (predictions['market_odds'] / predictions['model_odds'] - 1) * 100
            )
            value_bets = predictions[predictions['edge'] >= 5.0].copy()
        else:
            # No market odds - show top by probability
            value_bets = predictions.nlargest(10, 'win_probability').copy()
    
    # Sort by edge (or probability if no edge)
    sort_col = 'edge' if 'edge' in value_bets.columns else 'win_probability'
    value_bets = value_bets.sort_values(sort_col, ascending=False).head(10)
    
    # Build HTML
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 15px; }}
            th {{ background-color: #3498db; color: white; padding: 10px; text-align: left; }}
            td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .metric {{ background-color: #ecf0f1; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .footer {{ margin-top: 30px; color: #7f8c8d; font-size: 12px; }}
        </style>
    </head>
    <body>
        <h1>🏇 Daily Racing Predictions — {target_date.strftime('%A, %d %B %Y')}</h1>
        
        <div class="metric">
            <strong>Total Races:</strong> {predictions['race_id'].nunique() if 'race_id' in predictions.columns else len(predictions)} |
            <strong>Value Bets:</strong> {len(value_bets)} |
            <strong>Best Edge:</strong> {value_bets[sort_col].max():.1f}{'%' if sort_col == 'edge' else ''}
        </div>
        
        <h2>🎯 Top Value Bets</h2>
        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Course</th>
                    <th>Horse</th>
                    <th>Win %</th>
                    <th>Market Odds</th>
                    <th>Fair Odds</th>
                    <th>Edge %</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for _, row in value_bets.iterrows():
        race_time = row.get('race_time', 'TBC')
        course = row.get('course', 'Unknown')
        horse = row.get('horse_name', 'Unknown')
        win_prob = row['win_probability'] * 100
        market_odds = row.get('market_odds', '-')
        model_odds = row.get('model_odds', 1.0 / row['win_probability'] if row['win_probability'] > 0 else 999.0)
        edge = row.get('edge', 0.0)
        
        html += f"""
                <tr>
                    <td>{race_time}</td>
                    <td>{course}</td>
                    <td><strong>{horse}</strong></td>
                    <td>{win_prob:.1f}%</td>
                    <td>{market_odds if isinstance(market_odds, str) else f'{market_odds:.2f}'}</td>
                    <td>{model_odds:.2f}</td>
                    <td><strong>{edge:.1f}%</strong></td>
                </tr>
        """
    
    html += """
            </tbody>
        </table>
        
        <div class="footer">
            <p>This is an automated email from your Horse Racing Predictions system.</p>
            <p><em>Remember: Bet responsibly and within your limits. Past performance does not guarantee future results.</em></p>
        </div>
    </body>
    </html>
    """
    
    return html


def send_email(subject: str, html_body: str, dry_run: bool = False) -> bool:
    """Send email via SMTP."""
    
    sender = os.getenv("EMAIL_SENDER")
    recipient = os.getenv("EMAIL_RECIPIENT")
    password = os.getenv("EMAIL_PASSWORD")
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    
    if not all([sender, recipient, password]):
        print("[!] Missing email configuration in .env:")
        print("    Required: EMAIL_SENDER, EMAIL_RECIPIENT, EMAIL_PASSWORD")
        print("    Optional: SMTP_HOST (default: smtp.gmail.com), SMTP_PORT (default: 587)")
        return False
    
    if dry_run:
        print(f"[DRY RUN] Would send email:")
        print(f"  From: {sender}")
        print(f"  To: {recipient}")
        print(f"  Subject: {subject}")
        print(f"  Body length: {len(html_body)} chars")
        return True
    
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = recipient
        
        # Attach HTML
        html_part = MIMEText(html_body, 'html')
        msg.attach(html_part)
        
        # Send via SMTP
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        
        print(f"[✓] Email sent successfully to {recipient}")
        return True
    
    except Exception as e:
        print(f"[!] Failed to send email: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Send morning email with best bets")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD, default: today)")
    parser.add_argument("--test", action="store_true", help="Dry run without sending")
    
    args = parser.parse_args()
    
    # Determine target date
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"[!] Invalid date format: {args.date}. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        target_date = date.today()
    
    print(f"[*] Loading predictions for {target_date}")
    
    # Load predictions
    predictions = load_predictions(target_date)
    
    if predictions is None:
        print(f"[!] No predictions found for {target_date}")
        print(f"    Expected file: {DATA_DIR / f'predictions_{target_date}.csv'}")
        print(f"    Run: python scripts/predict_todays_races.py --date {target_date}")
        sys.exit(1)
    
    print(f"[✓] Loaded {len(predictions)} predictions")
    
    # Format email
    subject = f"🏇 Racing Predictions — {target_date.strftime('%d %b %Y')}"
    html_body = format_email_body(predictions, target_date)
    
    # Send
    success = send_email(subject, html_body, dry_run=args.test)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
