from locust import HttpUser, task, constant


class LoadTest(HttpUser):
    wait_time = constant(0)
    host = "http://localhost"

    @task
    def predict_batch_1(self):
        request_body = {"batches": [[1.0 for i in range(24)]]}
        self.client.post(
            "http://batch-1:80/predict", json=request_body, name="batch-1"
        )

    @task
    def predict_batch_32(self):
        request_body = {"batches": [[1.0 for i in range(24)] for i in range(32)]}
        self.client.post(
            "http://batch-32:80/predict", json=request_body, name="batch-32"
        )

    @task
    def predict_batch_64(self):
        request_body = {"batches": [[1.0 for i in range(24)] for i in range(64)]}
        self.client.post(
            "http://batch-64:80/predict", json=request_body, name="batch-64"
        )

    
