# Customer Churn Prediction

This project focuses on predicting customer churn using machine learning techniques. The goal is to identify customers who are likely to stop using a service, allowing businesses to take proactive measures to retain them.

## Project Structure

```
customer-churn-prediction/
├── data/               # Raw and processed data
├── notebooks/          # Jupyter notebooks for analysis
├── src/               # Source code
│   ├── data/          # Data processing scripts
│   ├── features/      # Feature engineering
│   ├── models/        # Model training and evaluation
│   └── visualization/ # Visualization utilities
├── models/            # Trained models
├── reports/           # Generated reports and visualizations
└── tests/             # Test files
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data preprocessing:
```bash
python src/data/preprocess.py
```

2. Feature engineering:
```bash
python src/features/build_features.py
```

3. Model training:
```bash
python src/models/train_model.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.