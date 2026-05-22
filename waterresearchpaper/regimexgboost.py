# ==========================================================
# CUSTOM ANFIS IMPLEMENTATION FOR ADSORPTION MODELING
# Compatible with Python 3.11
# Based on Jang (1993) and environmental engineering literature
# ==========================================================

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')


# ==========================================================
# 1. MEMBERSHIP FUNCTIONS
# ==========================================================
class GaussianMF:
    """Gaussian membership function: exp(-((x - c)^2) / (2 * sigma^2))"""

    def __init__(self, center, sigma):
        self.center = center
        self.sigma = sigma

    def compute(self, x):
        return np.exp(-0.5 * ((x - self.center) / (self.sigma + 1e-10)) ** 2)

    def update(self, dc, ds):
        self.center += dc
        self.sigma += ds


class BellMF:
    """Generalized Bell membership function"""

    def __init__(self, a, b, c):
        self.a = a  # width
        self.b = b  # slope
        self.c = c  # center

    def compute(self, x):
        return 1 / (1 + np.abs((x - self.c) / (self.a + 1e-10)) ** (2 * self.b))


# ==========================================================
# 2. ANFIS LAYER STRUCTURE
# ==========================================================
class ANFISLayer:
    """Base class for ANFIS layers"""

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class FuzzificationLayer(ANFISLayer):
    """Layer 1: Fuzzification - converts crisp inputs to fuzzy sets"""

    def __init__(self, n_inputs, n_mfs):
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.mfs = []

        # Initialize membership functions for each input
        for i in range(n_inputs):
            input_mfs = []
            for j in range(n_mfs):
                # Initialize with random parameters
                center = np.random.randn() * 0.5
                sigma = np.random.rand() * 0.5 + 0.5
                input_mfs.append(GaussianMF(center, sigma))
            self.mfs.append(input_mfs)

    def forward(self, x):
        """x shape: (batch_size, n_inputs)"""
        self.x = x
        batch_size = x.shape[0]
        self.mu = np.zeros((batch_size, self.n_inputs, self.n_mfs))

        for i in range(self.n_inputs):
            for j in range(self.n_mfs):
                self.mu[:, i, j] = self.mfs[i][j].compute(x[:, i])

        return self.mu


class RuleLayer(ANFISLayer):
    """Layer 2: Rule activation - computes firing strength of each rule"""

    def __init__(self, n_inputs, n_mfs):
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.n_rules = n_mfs ** n_inputs

    def forward(self, mu):
        """mu shape: (batch_size, n_inputs, n_mfs)"""
        self.mu = mu
        batch_size = mu.shape[0]

        # Generate all combinations of membership functions (rules)
        self.w = np.ones((batch_size, self.n_rules))

        rule_idx = 0
        indices = np.zeros(self.n_inputs, dtype=int)

        while rule_idx < self.n_rules:
            # Multiply membership values (AND operation using product t-norm)
            for b in range(batch_size):
                self.w[b, rule_idx] = np.prod([mu[b, i, indices[i]]
                                               for i in range(self.n_inputs)])

            # Increment indices
            indices[0] += 1
            for i in range(self.n_inputs - 1):
                if indices[i] >= self.n_mfs:
                    indices[i] = 0
                    indices[i + 1] += 1
                else:
                    break

            rule_idx += 1

        return self.w


class NormalizationLayer(ANFISLayer):
    """Layer 3: Normalization - normalizes firing strengths"""

    def forward(self, w):
        """w shape: (batch_size, n_rules)"""
        self.w = w
        self.w_sum = np.sum(w, axis=1, keepdims=True) + 1e-10
        self.w_norm = w / self.w_sum
        return self.w_norm


class DefuzzificationLayer(ANFISLayer):
    """Layer 4: Defuzzification - linear combination of inputs"""

    def __init__(self, n_inputs, n_rules):
        self.n_inputs = n_inputs
        self.n_rules = n_rules
        # Consequent parameters: for each rule, weight for each input + bias
        self.params = np.random.randn(n_rules, n_inputs + 1) * 0.1

    def forward(self, w_norm, x):
        """
        w_norm shape: (batch_size, n_rules)
        x shape: (batch_size, n_inputs)
        """
        self.w_norm = w_norm
        self.x = x
        batch_size = x.shape[0]

        # Add bias term to inputs
        x_aug = np.concatenate([x, np.ones((batch_size, 1))], axis=1)

        # Compute output for each rule
        self.f = np.zeros((batch_size, self.n_rules))
        for r in range(self.n_rules):
            self.f[:, r] = np.dot(x_aug, self.params[r])

        return self.f


class OutputLayer(ANFISLayer):
    """Layer 5: Summation - weighted average of rule outputs"""

    def forward(self, w_norm, f):
        """
        w_norm shape: (batch_size, n_rules)
        f shape: (batch_size, n_rules)
        """
        self.w_norm = w_norm
        self.f = f
        # Weighted sum
        self.y = np.sum(w_norm * f, axis=1)
        return self.y


# ==========================================================
# 3. ANFIS MODEL
# ==========================================================
class ANFIS:
    """
    Adaptive Neuro-Fuzzy Inference System

    Architecture:
    Layer 1: Fuzzification (Gaussian MFs)
    Layer 2: Rule activation (Product t-norm)
    Layer 3: Normalization
    Layer 4: Defuzzification (TSK consequents)
    Layer 5: Output (Weighted average)
    """

    def __init__(self, n_inputs, n_mfs=3, learning_rate=0.01):
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.n_rules = n_mfs ** n_inputs
        self.lr = learning_rate

        # Initialize layers
        self.layer1 = FuzzificationLayer(n_inputs, n_mfs)
        self.layer2 = RuleLayer(n_inputs, n_mfs)
        self.layer3 = NormalizationLayer()
        self.layer4 = DefuzzificationLayer(n_inputs, self.n_rules)
        self.layer5 = OutputLayer()

        self.trained = False

    def forward(self, x):
        """Forward pass through all layers"""
        # Layer 1: Fuzzification
        mu = self.layer1.forward(x)

        # Layer 2: Rule activation
        w = self.layer2.forward(mu)

        # Layer 3: Normalization
        w_norm = self.layer3.forward(w)

        # Layer 4: Defuzzification
        f = self.layer4.forward(w_norm, x)

        # Layer 5: Output
        y = self.layer5.forward(w_norm, f)

        return y

    def backward(self, x, y_true, y_pred):
        """Backward pass using hybrid learning (LSE + gradient descent)"""
        batch_size = x.shape[0]
        error = y_pred - y_true

        # Update consequent parameters using LSE
        w_norm = self.layer3.w_norm
        x_aug = np.concatenate([x, np.ones((batch_size, 1))], axis=1)

        for r in range(self.n_rules):
            # Weighted least squares for this rule
            W = np.diag(w_norm[:, r])
            A = W @ x_aug
            b = W @ y_true

            # Normal equation: (A^T A) theta = A^T b
            try:
                self.layer4.params[r] = np.linalg.lstsq(
                    A.T @ A + 1e-6 * np.eye(self.n_inputs + 1),
                    A.T @ b,
                    rcond=None
                )[0]
            except:
                pass

        # Update premise parameters using gradient descent
        for i in range(self.n_inputs):
            for j in range(self.n_mfs):
                mf = self.layer1.mfs[i][j]

                # Gradient computation
                grad_c = 0
                grad_s = 0

                for b in range(batch_size):
                    # Contribution of this MF to the error
                    mu_val = self.layer1.mu[b, i, j]

                    # Simplified gradient (approximate)
                    diff = (x[b, i] - mf.center) / (mf.sigma ** 2 + 1e-10)

                    grad_c += error[b] * mu_val * diff
                    grad_s += error[b] * mu_val * diff * (x[b, i] - mf.center) / (mf.sigma + 1e-10)

                # Update parameters
                mf.update(-self.lr * grad_c / batch_size,
                          -self.lr * grad_s / batch_size)

    def fit(self, X, y, epochs=100, batch_size=32, verbose=True):
        """Train ANFIS model"""
        n_samples = X.shape[0]
        n_batches = max(1, n_samples // batch_size)

        # Initialize consequent parameters using clustering
        kmeans = KMeans(n_clusters=min(self.n_rules, n_samples), random_state=42)
        kmeans.fit(X)

        history = {'loss': []}

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Compute loss
                loss = np.mean((y_pred - y_batch) ** 2)
                epoch_loss += loss

                # Backward pass
                self.backward(X_batch, y_batch, y_pred)

            avg_loss = epoch_loss / n_batches
            history['loss'].append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self.trained = True
        return history

    def predict(self, X):
        """Make predictions"""
        if not self.trained:
            print("Warning: Model not trained yet")
        return self.forward(X)


# ==========================================================
# 4. DATA PREPARATION (SAME AS YOUR CODE)
# ==========================================================
def prepare_data():
    df = pd.read_excel("data_2023(1).xlsx")
    df = df.drop(columns=["Unnamed: 12"], errors="ignore")
    df = df.rename(columns={"RE (%)": "RE"})

    numeric_cols = [
        "Surface area(m2/g)", "Adsorption capacity(mg/g)",
        "Intial Concentration(ppm)", "Contact Time (min.)",
        "Dose(g/L)", "RPM", "Initial pH", "T(K)", "RE"
    ]

    def clean_numeric(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip()
        if re.match(r"^\d+,\d+$", x):
            x = x.replace(",", ".")
        else:
            x = x.split(",")[0]
        x = x.replace("±", "").replace("~", "").replace("%", "")
        m = re.search(r"[-+]?\d*\.?\d+", x)
        return float(m.group()) if m else np.nan

    for c in numeric_cols:
        df[c] = df[c].apply(clean_numeric)

    df = df.dropna(subset=numeric_cols)
    df = df[(df["RE"] >= 0) & (df["RE"] <= 100)].reset_index(drop=True)

    # Add engineered features
    df["Site_Density"] = (df["Dose(g/L)"] * df["Surface area(m2/g)"]) / df["Intial Concentration(ppm)"]
    df["Cap_Load_Ratio"] = df["Adsorption capacity(mg/g)"] / df["Intial Concentration(ppm)"]
    df["LogTime"] = np.log1p(df["Contact Time (min.)"])
    df["Driving_Force"] = df["Intial Concentration(ppm)"] / (df["Dose(g/L)"] + 1e-6)
    df["Acidity_Strength"] = np.abs(df["Initial pH"] - 7)

    # Select only 4 most important features (based on domain knowledge & your LightGBM)
    # These are typically the most critical for adsorption:
    # 1. Initial Concentration - driving force
    # 2. Initial pH - surface charge and speciation
    # 3. Dose - available sites
    # 4. Contact Time (log-transformed) - kinetics
    feature_cols = [
        "Intial Concentration(ppm)",
        "Initial pH",
        "Dose(g/L)",
        "LogTime"
    ]

    X = df[feature_cols].values
    y = df["RE"].values

    # Normalize features (important for ANFIS)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    return X_norm, y, X_mean, X_std


# ==========================================================
# 5. MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ANFIS MODEL FOR HEAVY METAL ADSORPTION")
    print("=" * 60)

    # Load and prepare data
    X, y, X_mean, X_std = prepare_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nData shape: {X.shape}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Initialize and train ANFIS
    print("\n" + "-" * 60)
    print("Training ANFIS Model...")
    print("-" * 60)

    anfis = ANFIS(
        n_inputs=X_train.shape[1],
        n_mfs=2,  # Only 2 MFs per input: 2^4 = 16 rules (manageable!)
        learning_rate=0.01
    )

    history = anfis.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        verbose=True
    )

    # Evaluate
    print("\n" + "-" * 60)
    print("Evaluation Results")
    print("-" * 60)

    y_train_pred = np.clip(anfis.predict(X_train), 0, 100)
    y_test_pred = np.clip(anfis.predict(X_test), 0, 100)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"\nTraining R²: {train_r2:.4f}")
    print(f"Testing R²: {test_r2:.4f}")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Testing RMSE: {test_rmse:.4f}")
    print(f"\nNumber of fuzzy rules: {anfis.n_rules}")

    print("\n" + "=" * 60)
    print("ANFIS training complete!")
    print("=" * 60)