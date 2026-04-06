import requests
import time
import json
import os
from collections import deque
from datetime import datetime

PREDICT_URL        = "http://localhost:8000/predict"
WINDOW_SIZE        = 20    # number of recent predictions to track
DRIFT_THRESHOLD    = 0.3   # alert if positive rate shifts by 30%
BASELINE_POS_RATE  = 0.5   # expected positive rate from training data
LOG_FILE           = "./monitoring_log.json"

class DriftMonitor:
    def __init__(self, window_size=WINDOW_SIZE):
        self.window_size      = window_size
        self.predictions      = deque(maxlen=window_size)
        self.baseline_pos_rate = BASELINE_POS_RATE
        self.alerts           = []
        self.total_requests   = 0
        self.start_time       = datetime.now()

    def record_prediction(self, sentiment, confidence, latency_ms):
        """Record a prediction and check for drift"""
        self.total_requests += 1
        is_positive = 1 if sentiment == "POSITIVE" else 0
        self.predictions.append({
            "sentiment":   sentiment,
            "confidence":  confidence,
            "latency_ms":  latency_ms,
            "timestamp":   datetime.now().isoformat(),
            "is_positive": is_positive
        })

    def get_current_positive_rate(self):
        """Calculate positive prediction rate in current window"""
        if not self.predictions:
            return None
        return sum(p["is_positive"] for p in self.predictions) / len(self.predictions)

    def get_avg_confidence(self):
        """Calculate average confidence in current window"""
        if not self.predictions:
            return None
        return sum(p["confidence"] for p in self.predictions) / len(self.predictions)

    def get_avg_latency(self):
        """Calculate average latency in current window"""
        if not self.predictions:
            return None
        return sum(p["latency_ms"] for p in self.predictions) / len(self.predictions)

    def check_drift(self):
        """Check if prediction distribution has drifted"""
        if len(self.predictions) < self.window_size:
            return False, None

        current_pos_rate = self.get_current_positive_rate()
        drift            = abs(current_pos_rate - self.baseline_pos_rate)

        if drift > DRIFT_THRESHOLD:
            alert = {
                "type":              "PREDICTION_DRIFT",
                "timestamp":         datetime.now().isoformat(),
                "baseline_pos_rate": self.baseline_pos_rate,
                "current_pos_rate":  current_pos_rate,
                "drift":             drift,
                "threshold":         DRIFT_THRESHOLD,
                "message":           f"Positive rate shifted from {self.baseline_pos_rate:.0%} to {current_pos_rate:.0%}"
            }
            self.alerts.append(alert)
            return True, alert

        return False, None

    def check_latency_degradation(self):
        """Check if latency has increased significantly"""
        avg_latency = self.get_avg_latency()
        if avg_latency and avg_latency > 500:  # 500ms threshold
            alert = {
                "type":        "LATENCY_DEGRADATION",
                "timestamp":   datetime.now().isoformat(),
                "avg_latency": avg_latency,
                "threshold":   500,
                "message":     f"Average latency {avg_latency:.0f}ms exceeds 500ms threshold"
            }
            self.alerts.append(alert)
            return True, alert
        return False, None

    def print_stats(self):
        """Print current monitoring stats"""
        pos_rate    = self.get_current_positive_rate()
        avg_conf    = self.get_avg_confidence()
        avg_latency = self.get_avg_latency()

        print(f"\n📊 Monitoring Stats ({datetime.now().strftime('%H:%M:%S')})")
        print(f"   Total requests:    {self.total_requests}")
        print(f"   Window size:       {len(self.predictions)}/{self.window_size}")
        if pos_rate is not None:
            drift = abs(pos_rate - self.baseline_pos_rate)
            print(f"   Positive rate:     {pos_rate:.0%} (baseline: {self.baseline_pos_rate:.0%}, drift: {drift:.0%})")
        if avg_conf is not None:
            print(f"   Avg confidence:    {avg_conf:.2%}")
        if avg_latency is not None:
            print(f"   Avg latency:       {avg_latency:.0f}ms")
        print(f"   Alerts fired:      {len(self.alerts)}")

    def save_log(self):
        """Save monitoring log to disk"""
        log = {
            "start_time":     self.start_time.isoformat(),
            "total_requests": self.total_requests,
            "alerts":         self.alerts,
            "final_stats": {
                "positive_rate": self.get_current_positive_rate(),
                "avg_confidence": self.get_avg_confidence(),
                "avg_latency":    self.get_avg_latency(),
            }
        }
        with open(LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)
        print(f"\nMonitoring log saved to {LOG_FILE}")

def simulate_requests(monitor, num_requests=50):
    """Simulate incoming requests to test monitoring"""
    print(f"\nSimulating {num_requests} requests...")

    # Mix of reviews — mostly positive at first then shift to negative
    reviews = [
        # Normal distribution — 50/50
        "This product is absolutely amazing works perfectly",
        "Broke after two days complete waste of money",
        "Best purchase I have made this year highly recommend",
        "Terrible quality nothing like the description",
        "Great value for money very happy with purchase",
        "Stopped working after one week very disappointed",
        "Exceeded my expectations will buy again",
        "Poor quality do not waste your money",

        # Drift simulation — mostly negative (simulates product recall)
        "Product was recalled dangerous do not use",
        "Returned immediately defective out of box",
        "Worst purchase ever completely useless",
        "Dangerous product caused damage avoid at all costs",
        "False advertising nothing works as described",
    ]

    for i in range(num_requests):
        # Simulate drift after halfway point
        if i < num_requests // 2:
            # Normal traffic — balanced
            review = reviews[i % 8]
        else:
            # Drifted traffic — mostly negative
            review = reviews[8 + (i % 5)]

        try:
            response = requests.post(
                PREDICT_URL,
                json={"text": review},
                timeout=30
            )
            result = response.json()

            monitor.record_prediction(
                sentiment   = result["sentiment"],
                confidence  = result["confidence"],
                latency_ms  = result["latency_ms"]
            )

            # Check for drift every 5 requests
            if (i + 1) % 5 == 0:
                monitor.print_stats()

                drift_detected, alert = monitor.check_drift()
                if drift_detected:
                    print(f"\n🚨 DRIFT ALERT: {alert['message']}")
                    print(f"   Action: Trigger retraining pipeline")

                latency_issue, alert = monitor.check_latency_degradation()
                if latency_issue:
                    print(f"\n⚠️  LATENCY ALERT: {alert['message']}")

        except Exception as e:
            print(f"Request {i+1} failed: {e}")

        time.sleep(0.1)  # small delay between requests

def run_monitoring():
    print("\n" + "="*50)
    print("STAGE 4 — MONITORING")
    print("="*50)

    monitor = DriftMonitor(window_size=WINDOW_SIZE)

    print(f"Starting drift monitor")
    print(f"  Baseline positive rate: {BASELINE_POS_RATE:.0%}")
    print(f"  Drift threshold:        {DRIFT_THRESHOLD:.0%}")
    print(f"  Window size:            {WINDOW_SIZE} requests")

    # Simulate requests with drift
    simulate_requests(monitor, num_requests=50)

    # Final stats
    monitor.print_stats()
    monitor.save_log()

    # Summary
    if monitor.alerts:
        print(f"\n🚨 {len(monitor.alerts)} alerts fired during monitoring")
        for alert in monitor.alerts:
            print(f"   {alert['type']}: {alert['message']}")
        print(f"\n→ Recommendation: Retrain model with recent data")
        return True  # drift detected — retrain
    else:
        print(f"\n✅ No drift detected — model performing as expected")
        return False  # no drift — no retrain needed

if __name__ == "__main__":
    run_monitoring()