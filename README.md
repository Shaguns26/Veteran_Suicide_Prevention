# 🛟 Lifeline: Context-Aware Anomaly Detection for Veteran Suicide Prevention

**An Unsupervised Machine Learning system designed to detect behavioral anomalies in administrative healthcare data, enabling proactive mental health interventions for veterans.**

---

## 📌 Business Context & Social Impact

### **The Crisis**
Veteran suicide is a critical national challenge. Traditional prevention models rely on **Self-Reporting** (e.g., PHQ-9 depression surveys), which often fail due to:
* **Stigma:** Veterans may fear professional repercussions for admitting struggle.
* **Stoicism:** A cultural reluctance to seek help until a crisis is acute.
* **Access:** Many at-risk veterans are not actively seeing a therapist.

### **The Solution: "Digital Exhaust"**
Patients generate data patterns—billing codes, pharmacy logistics, and address changes—long before a behavioral crisis occurs.
**Lifeline** is a **Privacy-First Triage Tool** that flags high-risk behavioral anomalies using **Administrative Metadata** (not private therapy notes), alerting caseworkers to initiate "Buddy Check" calls.

---

## 🛠️ Technical Evolution (V1 vs. V3)

This project evolved from a basic statistical model to a context-aware AI system to solve the "Rural Paradox."

### **⚠️ V1: The "Naive" Model (Basic Isolation Forest)**
* **Approach:** Fed raw behavioral data (e.g., `home_delivery_ratio`, `er_visits`) directly into an Anomaly Detection algorithm.
* **The Failure ("The Rural Paradox"):** The model flagged **Rural Veterans** as "High Risk" simply because they used mail-order pharmacies (high isolation score).
* **The Insight:** For a veteran living 50 miles from a pharmacy, mail-order is **Logistical**, not **Behavioral**. The model was punishing geography, not risk.

### **✅ V2: The "Context-Aware" System (Current Version)**
* **Innovation:** We introduced **Contextual Z-Scoring**. Instead of raw numbers, we calculate **Deviations relative to the Peer Group**.
* **Technical Implementation:**
    ```python
    # We group veterans by Era (Vietnam vs. Post-9/11) and Geography (Urban vs. Rural)
    groups = df.groupby(['era', 'is_rural'])
    
    # Calculate Deviation from the Peer Group Mean
    df['rel_isolation'] = groups['isolation_score'].transform(lambda x: x - x.mean())
    ```
* **Result:** A rural veteran with high home delivery (matching their rural peers) is now scored as **Safe (0.0)**, whereas an urban veteran with the same behavior (deviating from urban peers) is scored as **Risk (+3.0)**.

---

## 📊 Data Strategy: The Simulation Engine

Due to HIPAA privacy laws, public datasets linking insurance claims to suicide risk do not exist. We engineered a **Probabilistic Simulation Engine** based on clinical risk factors verified by the Department of Veterans Affairs (VA).

**We simulate 10,000 Veterans across 3 "Tribes":**
1.  **Post-9/11 Combat Arms:** High risk of "Transition Stress" (first 24 months post-discharge).
2.  **Vietnam/Gulf Era:** High risk of "Polypharmacy" (Opioids + Benzodiazepines).
3.  **Peacetime Support:** Baseline control group.

### **Key "Red Flag" Features:**
* **The Transition Cliff:** Time since discharge < 24 months.
* **Crisis Care Ratio:** High Emergency Room usage / Low Primary Care usage.
* **Polypharmacy Risk:** Dangerous overlap of Pain Meds (Opioids) and Anxiety Meds (Benzos).
* **Social Withdrawal:** Exclusively using home delivery for prescriptions (relative to peer group).

---

## 🚀 Key Insights & Results

### **1. The Transition Cliff**
Our analysis confirms that risk is non-linear. Post-9/11 veterans show a massive spike in "Crisis Care" events in the **first 12-24 months** after leaving active duty, highlighting the need for targeted onboarding support.

### **2. Community Clustering (t-SNE)**
By projecting the data into 2D space, we proved that veterans exist in distinct "Tribes." Anomalies (Risk Cases) appear as outliers at the **edges** of these tribes, not as a single global cluster. This validates our "Community-Aware" modeling approach.

### **3. Performance (Recall Focus)**
* **Metric:** We optimize for **Recall** (Catching the Risk).
* **Trade-off:** We accept a higher False Positive rate (triggering a harmless phone call) to minimize False Negatives (missing a life-threatening crisis).

---

## 💻 How to Run

### **Prerequisites**
```bash
pip install pandas numpy scikit-learn seaborn matplotlib

```

### **Quick Start**

1. Open `Lifeline_Veteran_Anomaly_Detection.ipynb`.
2. Run the **Data Simulation** block to generate the 10,000 veteran profiles.
3. Run the **Contextual Feature Engineering** block to fix the "Rural Paradox."
4. Train the **Isolation Forest** and view the Risk Heatmap.

---

**Tech Stack:** Python, Scikit-Learn (Isolation Forest), Pandas, t-SNE

**Domain:** Healthcare AI, Anomaly Detection, Social Good

## 👤 Author

* **Shagun Sharma** - *Machine Learning Engineer*

**Graduate Student, Duke University - Fuqua School of Business**

  * [GitHub Profile](https://github.com/Shaguns26)
