"""
create_dataset.py
─────────────────
Generates a synthetic IT support-ticket dataset (300 rows) and
saves it to data/support_tickets.csv and to the HuggingFace Hub.

Labels (5 classes):
  0 → billing
  1 → hardware
  2 → network
  3 → account
  4 → software
"""

import csv
import random
import os

random.seed(42)

TEMPLATES = {
    "billing": [
        "I was charged twice for my subscription this month.",
        "My invoice shows incorrect pricing for the pro plan.",
        "I need a refund for the accidental upgrade charge.",
        "The payment failed but I was still billed.",
        "Can you explain the extra line item on my statement?",
        "My annual plan renewal was charged at the wrong rate.",
        "I cancelled before the billing cycle but was still charged.",
        "The promo discount wasn't applied to my latest invoice.",
        "I need to update my payment method but I'm getting an error.",
        "My credit card was charged without authorization.",
        "How do I download my tax invoice for last quarter?",
        "The enterprise plan was not activated after payment.",
        "I received a collection notice but my account is current.",
        "My billing address is wrong on all my invoices.",
        "I'm disputing a charge from three weeks ago.",
    ],
    "hardware": [
        "My laptop fan is making a loud grinding noise.",
        "The keyboard on my workstation is unresponsive.",
        "My monitor flickers when connected via HDMI.",
        "The USB ports on my docking station stopped working.",
        "My printer is showing an offline error even when powered on.",
        "The battery on my company laptop drains in under two hours.",
        "My trackpad stopped clicking after the latest OS update.",
        "The power adapter for my device sparks when plugged in.",
        "My external hard drive is not detected by the system.",
        "The webcam LED is on but no image appears in video calls.",
        "My desk phone handset has static on all calls.",
        "The touchscreen on the conference room display is unresponsive.",
        "My laptop screen has a large crack down the middle.",
        "The SD card slot on my device doesn't read any cards.",
        "The cooling pad I ordered is not compatible with my laptop model.",
    ],
    "network": [
        "I cannot connect to the VPN from home.",
        "The office Wi-Fi drops every 20 minutes on my floor.",
        "I'm getting DNS resolution errors for internal sites.",
        "My download speed dropped to under 1 Mbps today.",
        "I can't reach the shared network drive from branch office.",
        "The firewall is blocking my access to an approved tool.",
        "Ping times to our cloud servers have spiked to 400ms.",
        "I keep getting a proxy authentication error on Chrome.",
        "Remote Desktop stops connecting after a few seconds.",
        "The VoIP calls are dropping mid-conversation every few minutes.",
        "My device was assigned a duplicate IP address on the LAN.",
        "I cannot browse the internet but the LAN connection shows active.",
        "The SSL certificate for our internal portal expired.",
        "Network printing is unavailable from the second floor.",
        "I'm unable to join the company Teams meeting from the office network.",
    ],
    "account": [
        "I forgot my password and the reset email never arrived.",
        "My account was locked after too many failed login attempts.",
        "I need to transfer ownership of my account to a new employee.",
        "Two-factor authentication codes are not arriving via SMS.",
        "My username changed after the SSO migration and I can't log in.",
        "I need to disable MFA for an upcoming travel period.",
        "My account shows as suspended even though I renewed.",
        "I cannot change my email address in the profile settings.",
        "New hire's account was not provisioned before their start date.",
        "An ex-employee's account is still active and needs to be disabled.",
        "I'm getting a 'session expired' error immediately after logging in.",
        "My profile photo is not updating despite multiple attempts.",
        "The guest account I created for a vendor has too many permissions.",
        "I need a temporary password for a shared account.",
        "My access to the HR portal was revoked by mistake.",
    ],
    "software": [
        "The application crashes every time I try to export to PDF.",
        "I'm getting a 'missing DLL' error when launching the tool.",
        "The software update failed halfway and now the app won't open.",
        "My IDE keeps freezing when I open files larger than 5 MB.",
        "The plugin I installed yesterday is causing crashes.",
        "I need admin rights to install the new project management tool.",
        "The license for our design software expired unexpectedly.",
        "The macro I recorded in Excel is throwing a runtime error.",
        "AutoSave is not working in the latest version of the office suite.",
        "The antivirus software is quarantining a legitimate work file.",
        "I cannot uninstall the old version before installing the new one.",
        "The application is using 98% CPU at idle.",
        "Font rendering is blurry after upgrading to the new display drivers.",
        "The browser extension for our CRM stopped working after an update.",
        "I need to roll back the latest patch—it broke our workflow.",
    ],
}

LABEL_MAP = {name: idx for idx, name in enumerate(TEMPLATES.keys())}


def generate_dataset(n_per_class: int = 60) -> list[dict]:
    rows = []
    for label_name, templates in TEMPLATES.items():
        label_id = LABEL_MAP[label_name]
        for _ in range(n_per_class):
            base = random.choice(templates)
            # light augmentation: random prefix/suffix to vary samples
            prefixes = [
                "", "Hi, ", "Hello, ", "Good morning — ", "Hey there, ",
                "Urgent: ", "FYI — ", "Quick question: ",
            ]
            suffixes = [
                "", " Please help.", " Appreciate any guidance.",
                " This is affecting my work.", " Let me know ASAP.",
                " Thanks in advance.", " Regards.", "",
            ]
            text = random.choice(prefixes) + base + random.choice(suffixes)
            rows.append({"text": text.strip(), "label": label_id, "label_name": label_name})
    random.shuffle(rows)
    return rows


def save_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "label_name"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows → {path}")


if __name__ == "__main__":
    data = generate_dataset(n_per_class=60)  # 300 total
    save_csv(data, "data/support_tickets.csv")
    print("Label distribution:")
    from collections import Counter
    counts = Counter(r["label_name"] for r in data)
    for k, v in sorted(counts.items()):
        print(f"  {k:10s}: {v}")
