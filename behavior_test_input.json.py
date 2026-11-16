
{
  "samples": [
    {
      "timestamp": "2026-01-01T00:00:00Z",
      "agent_id": "host-01",
      "host": "test-host",
      "metrics": {
        "cpu_percent": 25.0,
        "memory_percent": 45.0,
        "disk_percent": 60.0,
        "load_1m": 1.0
      }
    },
    {
      "timestamp": "2026-01-01T00:01:00Z",
      "agent_id": "host-01",
      "host": "test-host",
      "metrics": {
        "cpu_percent": 97.0,
        "memory_percent": 92.0,
        "disk_percent": 60.0,
        "load_1m": 6.0
      }
    },
    {
      "timestamp": "2026-01-01T00:02:00Z",
      "agent_id": "host-01",
      "host": "test-host",
      "metrics": {
        "cpu_percent": 98.0,
        "memory_percent": 93.0,
        "disk_percent": 60.0,
        "load_1m": 6.5
      }
    }
  ],
  "security_events": [
    {
      "timestamp": "2026-01-01T00:02:00Z",
      "event_type": "brute_force_detected",
      "host": "test-host",
      "source": "test-security-feed"
    }
  ]
}

