# 🏠 California Housing Price Prediction – End-to-End ML Deployment

Docker Hub Link :- [https://hub.docker.com/repository/docker/atharvchaudharibits/california-housing-app](https://hub.docker.com/r/atharvchaudharibits/california-housing-app)

Demo Vedio Link :- https://drive.google.com/file/d/1IYoCIwmXfPI0LPHQFn48-Dmmtiz7Javw/view?usp=sharing

This project builds a complete, production-grade machine learning system for predicting California housing prices, covering data versioning, model tracking, API deployment, CI/CD, and monitoring.

---

## 📦 Repository and Data Versioning

- A dedicated GitHub repository is set up to manage the full ML lifecycle.
- The California housing dataset is loaded, cleaned, and preprocessed.
- Dataset versioning is implemented using DVC for reproducibility.
- The project follows a modular, scalable, and clean directory structure for clarity and maintainability.

---

## 🧠 Model Development & Experiment Tracking

- Multiple models were trained, including Linear Regression, Decision Tree, and Random Forest.
- MLflow is used to track all experiments, logging hyperparameters, metrics, and artifacts.
- The best-performing model is selected based on evaluation metrics and registered into the MLflow model registry.

---

## 🚀 API & Docker Packaging

- A FastAPI-based RESTful API is built to serve real-time housing price predictions.
- The service is containerized using Docker to ensure platform independence and smooth deployment.
- It accepts input via JSON and HTML forms and returns the predicted housing price as output.

---

## 🔁 CI/CD with GitHub Actions

- GitHub Actions workflows are configured to:
  - Automatically lint and test the code on every push.
  - Build and tag Docker images.
  - Push images to Docker Hub for deployment.
- Deployment can be done locally using Docker or scripted automation.

---

## 📊 Logging and Monitoring

- All incoming requests and model predictions are logged to both console and file.
- Structured emoji-based logging improves traceability and readability.
- A Prometheus-compatible `/metrics` endpoint is exposed for live monitoring.

---
