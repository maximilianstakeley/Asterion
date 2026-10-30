# In telemetry_collector.py, replace _mount_for_path() and the "filesystem"
# field in _collect_disk() with this implementation.

@staticmethod
def _filesystem_identity(path: str) -> dict[str, str | None]:
    """Return real mount/device/type metadata or explicit unknowns."""
    try:
        resolved_path = Path(path).resolve()
        matches = []
        for partition in psutil.disk_partitions(all=True):
            try:
                mount_path = Path(partition.mountpoint).resolve()
                resolved_path.relative_to(mount_path)
                matches.append((len(str(mount_path)), partition, mount_path))
            except (ValueError, OSError, RuntimeError):
                continue

        if not matches:
            return {
                "mount_point": None,
                "device": None,
                "filesystem_type": None,
            }

        _, partition, mount_path = max(matches, key=lambda item: item[0])
        return {
            "mount_point": str(mount_path),
            "device": partition.device or None,
            "filesystem_type": partition.fstype or None,
        }
    except Exception:
        return {
            "mount_point": None,
            "device": None,
            "filesystem_type": None,
        }


# In _collect_disk(), after calculating percent, use:
identity = self._filesystem_identity(requested_path)
return {
    "percent": float(percent),
    "total_bytes": int(usage.total),
    "used_bytes": int(usage.used),
    "free_bytes": int(usage.free),
    "requested_path": requested_path,
    **identity,
}
