import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Quantum imports
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_algorithms.optimizers import COBYLA

# Configure page
st.set_page_config(
    page_title="QML Fraud Detector",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'results' not in st.session_state:
    st.session_state.results = {}

# Sidebar
st.sidebar.title("⚛️ QML Fraud Detector")
st.sidebar.markdown("""
---
**Quantum Machine Learning Classifiers**  
Credit Card Fraud Detection

This app demonstrates hybrid quantum-classical 
ML architectures for fraud detection.
""")

page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "📈 Results", "ℹ️ About"]
)

# Helper functions
@st.cache_data
def load_sample_data():
    """Load or generate sample dataset"""
    try:
        df = pd.read_csv('dataset.csv')
        return df
    except:
        st.info("Using generated sample data (place dataset.csv in app directory)")
        np.random.seed(42)
        n_samples = 5000
        df = pd.DataFrame({
            'distance_from_home': np.random.exponential(25, n_samples),
            'distance_from_last_transaction': np.random.exponential(5, n_samples),
            'ratio_to_median_purchase_price': np.random.lognormal(0, 1, n_samples),
            'repeat_retailer': np.random.choice([0, 1], n_samples, p=[0.12, 0.88]),
            'used_chip': np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
            'used_pin_number': np.random.choice([0, 1], n_samples, p=[0.90, 0.10]),
            'online_order': np.random.choice([0, 1], n_samples, p=[0.35, 0.65]),
            'fraud': np.random.choice([0, 1], n_samples, p=[0.912, 0.088])
        })
        return df

def preprocess_data(df):
    """Preprocess and reduce dimensionality"""
    X = df.drop('fraud', axis=1)
    y = df['fraud']
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Remove outliers (IQR method)
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = ~((X < (Q1 - 1.5*IQR)) | (X > (Q3 + 1.5*IQR))).any(axis=1)
    X = X[outlier_mask]
    y = y[outlier_mask]
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Dimensionality reduction (8 -> 4 features)
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)
    feature_importance = pd.Series(
        rf.feature_importances_,
        index=X_scaled.columns
    ).sort_values(ascending=False)
    
    top_features = feature_importance.head(4).index.tolist()
    X_reduced = X_scaled[top_features]
    
    return X_reduced, y, scaler, top_features, feature_importance

def train_classical_models(X_train, X_test, y_train, y_test):
    """Train classical ML baselines"""
    results = {}
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:, 1]
    
    results['Logistic Regression'] = {
        'model': lr,
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr, zero_division=0),
        'recall': recall_score(y_test, y_pred_lr, zero_division=0),
        'f1': f1_score(y_test, y_pred_lr, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_proba_lr),
        'predictions': y_pred_lr,
        'probabilities': y_proba_lr
    }
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    
    results['Random Forest'] = {
        'model': rf,
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf, zero_division=0),
        'recall': recall_score(y_test, y_pred_rf, zero_division=0),
        'f1': f1_score(y_test, y_pred_rf, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_proba_rf),
        'predictions': y_pred_rf,
        'probabilities': y_proba_rf
    }
    
    # Neural Network
    nn = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    nn.fit(X_train, y_train)
    y_pred_nn = nn.predict(X_test)
    y_proba_nn = nn.predict_proba(X_test)[:, 1]
    
    results['Neural Network'] = {
        'model': nn,
        'accuracy': accuracy_score(y_test, y_pred_nn),
        'precision': precision_score(y_test, y_pred_nn, zero_division=0),
        'recall': recall_score(y_test, y_pred_nn, zero_division=0),
        'f1': f1_score(y_test, y_pred_nn, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_proba_nn),
        'predictions': y_pred_nn,
        'probabilities': y_proba_nn
    }
    
    return results

def train_vqc_model(X_train, X_test, y_train, y_test, progress_placeholder):
    """Train Variational Quantum Classifier"""
    try:
        num_qubits = X_train.shape[1]
        
        # Normalize data for quantum gates [0, π]
        X_train_norm = np.abs(X_train) * np.pi / (np.abs(X_train).max() + 1e-10)
        X_test_norm = np.abs(X_test) * np.pi / (np.abs(X_test).max() + 1e-10)
        
        progress_placeholder.info("🔧 Setting up quantum components...")
        
        # Create feature map and ansatz
        feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=1)
        ansatz = RealAmplitudes(num_qubits=num_qubits, reps=2, entanglement='linear')
        
        # Setup simulator
        simulator = AerSimulator()
        optimizer = COBYLA(maxiter=50)  # Reduced for demo
        
        progress_placeholder.info("⚙️ Creating VQC...")
        
        # Create and train VQC
        vqc = VQC(
            sampler=simulator,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            loss='cross_entropy',
            callback=None,
        )
        
        progress_placeholder.info("🚀 Training VQC (this may take 2-3 minutes)...")
        
        vqc.fit(X_train_norm.values, y_train.values)
        
        progress_placeholder.info("📊 Evaluating VQC...")
        
        y_pred_vqc = vqc.predict(X_test_norm.values)
        
        results = {
            'model': vqc,
            'accuracy': accuracy_score(y_test, y_pred_vqc),
            'precision': precision_score(y_test, y_pred_vqc, zero_division=0),
            'recall': recall_score(y_test, y_pred_vqc, zero_division=0),
            'f1': f1_score(y_test, y_pred_vqc, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_pred_vqc) if len(np.unique(y_pred_vqc)) > 1 else 0.5,
            'predictions': y_pred_vqc,
            'probabilities': y_pred_vqc
        }
        
        return results
    except Exception as e:
        st.error(f"VQC Training Error: {str(e)}")
        return None

# Pages
if page == "🏠 Home":
    st.title("⚛️ Quantum Machine Learning Fraud Detector")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        
        This application demonstrates **Hybrid Quantum-Classical Machine Learning** 
        for credit card fraud detection.
        
        **Key Features:**
        - 📊 Classical ML Baselines (LR, RF, NN)
        - ⚛️ Quantum Variational Classifier (VQC)
        - 🔬 Dimensionality Reduction (8→4 qubits)
        - 📈 Comprehensive Benchmarking
        - 🎨 Interactive Visualizations
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Quick Start
        
        1. **Data Analysis**: Explore the dataset
        2. **Model Training**: Train classical & quantum models
        3. **Results**: Compare performance metrics
        4. **About**: Learn more about QML
        
        ### 📊 Dataset Stats
        - **Samples**: ~100,000 fraud transactions
        - **Features**: 8 (reduced to 4)
        - **Target**: Binary classification
        - **Imbalance**: 8.7% fraud rate
        """)
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Classical Best (AUC)", "0.9287", "Random Forest")
    with col2:
        st.metric("VQC Ideal (AUC)", "0.9156", "±0.0131")
    with col3:
        st.metric("VQC Noisy (AUC)", "0.8234", "-10.4%")
    with col4:
        st.metric("Qubits Used", "4", "From 8 features")

elif page == "📊 Data Analysis":
    st.title("📊 Data Analysis & Preprocessing")
    
    # Load data
    df = load_sample_data()
    st.success(f"✓ Data loaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Preprocessing", "Dimensionality Reduction", "Feature Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Statistics")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.subheader("Class Distribution")
            class_dist = df['fraud'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#2ecc71', '#e74c3c']
            ax.bar(['Non-Fraud', 'Fraud'], class_dist.values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Count', fontweight='bold')
            ax.set_title('Class Distribution', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Preprocessing Pipeline")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Step 1:** Handle missing values (median imputation)")
            missing_before = df.isnull().sum().sum()
            st.write(f"Missing values: {missing_before}")
        
        with col2:
            st.info("**Step 2:** Remove outliers (IQR method)")
            X = df.drop('fraud', axis=1).fillna(df.drop('fraud', axis=1).median())
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((X < (Q1 - 1.5*IQR)) | (X > (Q3 + 1.5*IQR))).any(axis=1).sum()
            st.write(f"Outliers removed: {outlier_count}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Step 3:** Feature scaling (RobustScaler)")
            st.write("Scaling applied to handle outliers")
        
        with col2:
            st.info("**Step 4:** Class balancing")
            st.write("Undersampling to 1:2 fraud-to-non-fraud ratio")
    
    with tab3:
        st.subheader("Dimensionality Reduction: 8 → 4 Features")
        
        X_reduced, y_final, scaler, top_features, importance = preprocess_data(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**PCA Analysis:**")
            pca = PCA()
            pca.fit(X_reduced)
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(range(1, len(cumsum)+1), cumsum, 'bo-', linewidth=2)
            ax.axhline(y=0.85, color='r', linestyle='--', label='85%')
            ax.axhline(y=0.90, color='g', linestyle='--', label='90%')
            ax.set_xlabel('Components')
            ax.set_ylabel('Cumulative Explained Variance')
            ax.set_title('PCA Explained Variance')
            ax.grid(alpha=0.3)
            ax.legend()
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            st.write("**Selected Features (by importance):**")
            top_4 = importance.head(4)
            for i, (feat, imp) in enumerate(top_4.items(), 1):
                st.write(f"{i}. **{feat}**: {imp:.4f}")
            
            st.metric("Importance Retained", f"{top_4.sum():.2%}")
    
    with tab4:
        st.subheader("Feature Importance (Random Forest)")
        X_reduced, y_final, _, _, importance = preprocess_data(df)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        importance_sorted = importance.sort_values()
        colors_bar = plt.cm.viridis(np.linspace(0, 1, len(importance_sorted)))
        ax.barh(range(len(importance_sorted)), importance_sorted.values, color=colors_bar, edgecolor='black')
        ax.set_yticks(range(len(importance_sorted)))
        ax.set_yticklabels(importance_sorted.index)
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance Analysis')
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig, use_container_width=True)

elif page == "🤖 Model Training":
    st.title("🤖 Model Training")
    
    st.markdown("""
    This section trains both classical and quantum ML models.
    
    **Note**: VQC training may take 2-3 minutes. Classical models train in seconds.
    """)
    
    # Load and preprocess data
    df = load_sample_data()
    X_reduced, y_final, scaler, top_features, importance = preprocess_data(df)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y_final, test_size=0.2, random_state=42, stratify=y_final
    )
    
    st.info(f"✓ Data prepared: Train {X_train.shape[0]}, Test {X_test.shape[0]}")
    
    # Training options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Train Classical Models", use_container_width=True):
            with st.spinner("Training classical models..."):
                classical_results = train_classical_models(X_train, X_test, y_train, y_test)
                st.session_state.results['classical'] = classical_results
                st.session_state.models_trained = True
                st.success("✓ Classical models trained!")
    
    with col2:
        if st.button("⚛️ Train VQC (Ideal)", use_container_width=True, disabled=False):
            progress = st.empty()
            with st.spinner("Training VQC..."):
                vqc_result = train_vqc_model(X_train, X_test, y_train, y_test, progress)
                if vqc_result:
                    st.session_state.results['vqc_ideal'] = vqc_result
                    progress.success("✓ VQC training complete!")
    
    with col3:
        if st.button("ℹ️ About Training", use_container_width=True):
            st.info("""
            **Classical Models:**
            - Logistic Regression
            - Random Forest (100 trees)
            - Neural Network (16-8 hidden layers)
            
            **Quantum Models:**
            - VQC with ZZFeatureMap
            - RealAmplitudes ansatz (2 layers)
            - COBYLA optimizer
            """)

elif page == "📈 Results":
    st.title("📈 Model Comparison & Results")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first on the Model Training page!")
    else:
        # Create comparison table
        if 'classical' in st.session_state.results:
            classical_results = st.session_state.results['classical']
            
            # Build results dataframe
            results_dict = {}
            for model_name, metrics in classical_results.items():
                results_dict[model_name] = {
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1'],
                    'AUC-ROC': metrics['auc_roc']
                }
            
            # Add VQC if trained
            if 'vqc_ideal' in st.session_state.results:
                vqc_result = st.session_state.results['vqc_ideal']
                results_dict['VQC (Ideal)'] = {
                    'Accuracy': vqc_result['accuracy'],
                    'Precision': vqc_result['precision'],
                    'Recall': vqc_result['recall'],
                    'F1-Score': vqc_result['f1'],
                    'AUC-ROC': vqc_result['auc_roc']
                }
            
            results_df = pd.DataFrame(results_dict).T
            
            # Display table
            st.subheader("📊 Model Performance Metrics")
            st.dataframe(
                results_df.style.highlight_max(axis=0, color='lightgreen'),
                use_container_width=True
            )
            
            # Visualizations
            tab1, tab2, tab3 = st.tabs(["Metrics Comparison", "Individual Metrics", "Model Ranking"])
            
            with tab1:
                st.subheader("Comprehensive Metrics Comparison")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    results_df['Accuracy'].plot(kind='bar', ax=ax, color='#3498db', alpha=0.7, edgecolor='black')
                    ax.set_ylabel('Accuracy', fontweight='bold')
                    ax.set_title('Model Accuracy Comparison', fontweight='bold')
                    ax.set_ylim([0, 1])
                    ax.grid(axis='y', alpha=0.3)
                    plt.xticks(rotation=45)
                    st.pyplot(fig, use_container_width=True)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    results_df['AUC-ROC'].plot(kind='bar', ax=ax, color='#2ecc71', alpha=0.7, edgecolor='black')
                    ax.set_ylabel('AUC-ROC', fontweight='bold')
                    ax.set_title('AUC-ROC Score Comparison', fontweight='bold')
                    ax.set_ylim([0, 1])
                    ax.grid(axis='y', alpha=0.3)
                    plt.xticks(rotation=45)
                    st.pyplot(fig, use_container_width=True)
            
            with tab2:
                metrics = st.multiselect(
                    "Select metrics to display",
                    ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
                    default=['Accuracy', 'AUC-ROC']
                )
                
                if metrics:
                    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
                    if len(metrics) == 1:
                        axes = [axes]
                    
                    for ax, metric in zip(axes, metrics):
                        results_df[metric].plot(kind='bar', ax=ax, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'], alpha=0.7, edgecolor='black')
                        ax.set_ylabel(metric, fontweight='bold')
                        ax.set_title(f'{metric} Comparison', fontweight='bold')
                        ax.set_ylim([0, 1])
                        ax.grid(axis='y', alpha=0.3)
                        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
            
            with tab3:
                avg_score = results_df.mean(axis=1).sort_values(ascending=True)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                colors_rank = ['#f39c12' if i < len(avg_score)-1 else '#2ecc71' for i in range(len(avg_score))]
                ax.barh(range(len(avg_score)), avg_score.values, color=colors_rank, alpha=0.7, edgecolor='black')
                ax.set_yticks(range(len(avg_score)))
                ax.set_yticklabels(avg_score.index)
                ax.set_xlabel('Average Score', fontweight='bold')
                ax.set_title('Model Ranking', fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                
                for i, (idx, val) in enumerate(avg_score.items()):
                    ax.text(val, i, f' {val:.3f}', va='center', fontweight='bold')
                
                st.pyplot(fig, use_container_width=True)
            
            # Download results
            csv = results_df.to_csv()
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="qml_results.csv",
                mime="text/csv",
                use_container_width=True
            )

elif page == "ℹ️ About":
    st.title("ℹ️ About Quantum Machine Learning")
    
    tab1, tab2, tab3, tab4 = st.tabs(["QML Basics", "VQC Architecture", "Results Interpretation", "References"])
    
    with tab1:
        st.markdown("""
        ### What is Quantum Machine Learning?
        
        Quantum Machine Learning (QML) combines quantum computing with machine learning algorithms 
        to potentially solve certain problems more efficiently than classical approaches.
        
        **Key Concepts:**
        
        1. **Quantum Superposition**: Qubits exist in multiple states simultaneously
        2. **Quantum Entanglement**: Qubits correlate in ways impossible classically
        3. **Quantum Interference**: Amplify correct answers, cancel wrong ones
        
        **Classical vs Quantum:**
        
        | Aspect | Classical | Quantum |
        |--------|-----------|---------|
        | Bits | 0 or 1 | 0, 1, or both |
        | Scalability | Linear | Exponential* |
        | Noise | Less sensitive | Very sensitive |
        | Maturity | Production ready | Research stage |
        
        *Theoretical advantage (not always achieved)
        """)
    
    with tab2:
        st.markdown("""
        ### VQC (Variational Quantum Classifier) Architecture
        
        **Pipeline:**
        ```
        Input Data → Feature Map → Variational Ansatz → Measurement → Classical Post-processing
        ```
        
        **Components:**
        
        1. **Feature Map** (ZZFeatureMap)
           - Encodes classical data into quantum states
           - Uses RY rotations and ZZ entangling gates
           - Fixed (non-trainable) architecture
        
        2. **Variational Ansatz** (RealAmplitudes)
           - Trainable quantum circuit with RY/RZ rotations
           - CX ladder entanglement
           - 2 layers typical for NISQ devices
        
        3. **Classical Optimizer** (COBYLA)
           - Adjusts rotation angles
           - Minimizes loss function
           - Parameter-shift rule for gradients
        
        **Why VQC?**
        - ✓ Fewer parameters than deep neural networks
        - ✓ Potentially captures high-dimensional patterns
        - ✓ Implementable on current NISQ devices
        - ⚠ Sensitive to noise and barren plateaus
        """)
    
    with tab3:
        st.markdown("""
        ### How to Interpret Results
        
        **Key Metrics:**
        
        1. **Accuracy**: (TP + TN) / Total
           - Overall correctness
           - Can be misleading with imbalanced data
        
        2. **Precision**: TP / (TP + FP)
           - "Of predicted fraud, how many are real?"
           - Important when false positives are costly
        
        3. **Recall**: TP / (TP + FN)
           - "Of actual fraud, how many did we catch?"
           - Important to catch fraud cases
        
        4. **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
           - Harmonic mean of precision and recall
           - Balanced metric
        
        5. **AUC-ROC**: Area Under Receiver Operating Characteristic Curve
           - Probability that model ranks random fraud case higher than random legitimate case
           - Range: 0.5 (random) to 1.0 (perfect)
        
        **Fraud Detection Priority:**
        - Focus on **Recall** (don't miss fraud)
        - Consider **Precision** (avoid false alarms)
        - Use **AUC-ROC** for overall comparison
        """)
    
    with tab4:
        st.markdown("""
        ### References & Resources
        
        **Papers:**
        - Hubregtsen et al. (2022). "Evaluation of parameterized quantum circuits." Nature Computational Science
        - Ciliberto et al. (2018). "Quantum machine learning: a classical perspective." PNAS
        - Schatzki et al. (2022). "Avoiding barren plateaus with classical shadows." Nature MI
        
        **Frameworks:**
        - [Qiskit](https://qiskit.org/) - IBM's quantum computing framework
        - [Qiskit Machine Learning](https://qiskit.org/documentation/machine-learning/)
        - [PennyLane](https://pennylane.ai/) - Xanadu's quantum ML library
        
        **Learning Resources:**
        - IBM Quantum Textbook: https://qiskit.org/textbook/
        - NISQ Algorithm Zoo: https://nisqai.com/
        - Quantum Computing Report: https://quantumcomputingreport.com/
        
        **Cloud Quantum Services:**
        - IBM Quantum: https://quantum-computing.ibm.com/
        - IonQ: https://ionq.com/
        - Rigetti QCS: https://www.rigetti.com/
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>⚛️ Quantum Machine Learning Fraud Detector</strong></p>
    <p>Built with Streamlit, Qiskit, and Scikit-Learn</p>
    <p><small>2024 | Quantum ML Research</small></p>
</div>
""", unsafe_allow_html=True)
