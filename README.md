# FedEx DCA Management: Digital & AI Transformation
> **Status:** FedEx SMART Hackathon Prototype | **Category:** AI & Data Visualization

## Overview
This repository contains the prototype for an intelligent Debt Collection Agency (DCA) Management platform designed to streamline recovery operations, improve transparency, and automate case prioritization. 

The platform transitions FedEx from manual, spreadsheet-based tracking to a centralized digital ecosystem. It integrates legacy data with predictive analytics to optimize debt recovery workflows.

## Installation and Setup
To run this project locally and explore the dashboard:

1. **Clone the repository:**
   `git clone https://github.com/your-username/your-repo-name.git`

2. **Install dependencies:**
   `pip install -r requirements.txt`

3. **Run the application:**
   `streamlit run app.py`

## Key Technical Components
* **AI Logic:** Implements a Random Forest Classifier to score accounts based on recovery probability.
* **Automated Workflow:** Replaces manual case allocation with a data-driven prioritization matrix.
* **Real-time Analytics:** Built with Streamlit and Plotly Express to provide executive-level visibility.
* **Data Pipeline:**
    * **Preprocessing:** Handling categorical encoding and scaling for debt aging buckets using Pandas.
    * **Inference:** Generating real-time probability scores for account prioritization.

## Technical Stack
* **Language:** Python 3.x
* **UI Framework:** Streamlit
* **Data Science:** Scikit-Learn, Pandas, NumPy
* **Visualization:** Plotly Express

## License
This project is licensed under the MIT License.
