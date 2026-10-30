# In TelemetryCollector.collect(), replace the existing all_ok calculation
# and collection_status expression with this block.
statuses = self._record_status(metric_records)
metric_states = [item["status"] for item in statuses.values()]

if metric_states and all(state == "ok" for state in metric_states):
    collection_status = "ok"
elif metric_states and all(state == "unavailable" for state in metric_states):
    collection_status = "failed"
else:
    collection_status = "partial"

# Then use:
return {
    "agent_id": self.agent_id,
    "host": self.host_id,
    "timestamp": observation_timestamp,
    "collection_status": collection_status,
    "metrics": metrics,
    "metric_status": statuses,
}
