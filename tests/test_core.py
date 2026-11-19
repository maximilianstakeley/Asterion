import unittest
from Asterion.detection import detect_anomalies
from Asterion.rootcause import rank_hypotheses
class CoreTests(unittest.TestCase):
    def test_high_cpu_detected(self):
        result=detect_anomalies({"cpu_percent":96,"memory_percent":20,"disk_percent":30,"network_connections":2})
        self.assertTrue(any(x["category"]=="high_cpu" for x in result))
    def test_normal_metrics(self):
        self.assertEqual(detect_anomalies({"cpu_percent":20,"memory_percent":30,"disk_percent":40,"network_connections":10}),[])
    def test_hypotheses_have_evidence(self):
        result=rank_hypotheses({},[{"category":"high_memory"}],{"available":False})
        self.assertTrue(result and result[0]["evidence"])
if __name__=="__main__": unittest.main()
