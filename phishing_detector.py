
import pandas as pd
import numpy as np
import re
import tldextract
import urllib.parse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys
import time

class PhishingURLDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def extract_features(self, url):
        """Extract features from a URL"""
        features = {}
        
        # Parse the URL
        parsed = urllib.parse.urlparse(url)
        extract = tldextract.extract(url)
        
        # Length-based features
        features['url_length'] = len(url)
        features['domain_length'] = len(extract.domain)
        features['path_length'] = len(parsed.path)
        
        # Count-based features
        features['dot_count'] = url.count('.')
        features['hyphen_count'] = url.count('-')
        features['underscore_count'] = url.count('_')
        features['slash_count'] = url.count('/')
        features['question_count'] = url.count('?')
        features['equal_count'] = url.count('=')
        features['at_count'] = url.count('@')
        features['and_count'] = url.count('&')
        features['exclamation_count'] = url.count('!')
        features['space_count'] = url.count(' ')
        features['tilde_count'] = url.count('~')
        features['comma_count'] = url.count(',')
        features['plus_count'] = url.count('+')
        features['asterisk_count'] = url.count('*')
        features['hash_count'] = url.count('#')
        features['dollar_count'] = url.count('$')
        features['percent_count'] = url.count('%')
        
        # Binary features
        features['is_https'] = 1 if parsed.scheme == 'https' else 0
        features['has_ip_address'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0
        features['has_at_symbol'] = 1 if '@' in url else 0
        features['has_double_slash'] = 1 if '//' in parsed.path else 0
        features['has_hex'] = 1 if re.search(r'%[0-9A-Fa-f]{2}', url) else 0
        features['has_www'] = 1 if extract.subdomain == 'www' else 0
        
        # Domain-specific features
        features['domain_in_path'] = 1 if extract.domain in parsed.path.lower() else 0
        features['domain_in_subdomain'] = 1 if extract.domain in extract.subdomain.lower() else 0
        features['suspicious_tld'] = 1 if extract.suffix in ['xyz', 'tk', 'ml', 'ga', 'cf', 'info', 'online', 'site'] else 0
        
        # Word-based features
        suspicious_words = ['secure', 'account', 'webscr', 'login', 'ebayisapi', 'sign', 'banking', 'confirm']
        features['suspicious_words'] = sum(1 for word in suspicious_words if word in url.lower())
        
        return features
    
    def train(self, training_data_path):
        """Train the phishing detection model"""
        # Load dataset (CSV with 'url' and 'is_phishing' columns)
        print("Loading dataset...")
        df = pd.read_csv(training_data_path)
        
        # Extract features
        print("Extracting features...")
        X = []
        for url in df['url']:
            features = self.extract_features(url)
            X.append(list(features.values()))
        
        X = np.array(X)
        y = df['is_phishing'].values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        print("Training model...")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        print("Evaluating model...")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(report)
        print("Confusion Matrix:")
        print(conf_matrix)
        
        return accuracy, report, conf_matrix
    
    def save_model(self, model_path, scaler_path):
        """Save the trained model and scaler"""
        if self.model is None:
            raise Exception("Model not trained yet!")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path, scaler_path):
        """Load a trained model and scaler"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print("Model and scaler loaded successfully!")
    
    def predict(self, url):
        """Predict whether a URL is phishing or not"""
        if self.model is None:
            raise Exception("Model not trained or loaded yet!")
        
        features = self.extract_features(url)
        X = np.array(list(features.values())).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0][1]
        
        return {
            'url': url,
            'is_phishing': bool(prediction),
            'probability': probability,
            'features': features
        }

def generate_sample_dataset(output_path, size=1000):
    """Generate a sample dataset for demo purposes"""
    print("Generating sample phishing and legitimate URLs...")
    urls = []
    labels = []
    
    # Sample legitimate URLs
    legitimate_domains = [
        "google.com", "facebook.com", "amazon.com", "youtube.com", 
        "twitter.com", "instagram.com", "linkedin.com", "microsoft.com",
        "apple.com", "github.com", "stackoverflow.com", "wikipedia.org"
    ]
    
    paths = [
        "", "/index.html", "/about", "/contact", "/products", "/services",
        "/login", "/signup", "/profile", "/settings", "/help", "/faq"
    ]
    
    params = [
        "", "?id=123", "?page=1", "?search=example", "?lang=en",
        "?ref=homepage", "?source=direct", "?utm_source=google"
    ]
    
    # Generate legitimate URLs
    for i in range(size // 2):
        domain = np.random.choice(legitimate_domains)
        path = np.random.choice(paths)
        param = np.random.choice(params)
        protocol = "https" if np.random.random() > 0.1 else "http"
        
        url = f"{protocol}://{domain}{path}{param}"
        urls.append(url)
        labels.append(0)  # 0 for legitimate
    
    # Generate phishing URLs
    for i in range(size // 2):
        is_secure = np.random.random() > 0.7
        protocol = "https" if is_secure else "http"
        
        # Use patterns common in phishing URLs
        if np.random.random() > 0.5:
            # Brand name in subdomain of suspicious domain
            brand = np.random.choice(["paypal", "amazon", "apple", "microsoft", "facebook"])
            suspicious_domain = np.random.choice(["secure-verify.com", "account-verify.net", "login-secure.org",
                                                 "verification-center.com", "secure-login.info"])
            url = f"{protocol}://{brand}.{suspicious_domain}/login.html"
        else:
            # Long domain with suspicious words
            suspicious_words = ["secure", "verify", "login", "account", "update", "confirm"]
            suspicious_word = np.random.choice(suspicious_words)
            brand = np.random.choice(["paypal", "amazon", "apple", "microsoft", "facebook"])
            tld = np.random.choice(["com", "net", "org", "info", "xyz", "online"])
            
            url = f"{protocol}://{brand}-{suspicious_word}.{tld}/login.php?secure=true&redirect={np.random.randint(10000, 99999)}"
            
            # Add some more complexity sometimes
            if np.random.random() > 0.7:
                url += f"&session={np.random.randint(1000000, 9999999)}"
                
            # Add IP address sometimes
            if np.random.random() > 0.8:
                ip = f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
                url = f"{protocol}://{ip}/~{brand}/"
        
        urls.append(url)
        labels.append(1)  # 1 for phishing
    
    # Create DataFrame and shuffle
    df = pd.DataFrame({
        'url': urls,
        'is_phishing': labels
    })
    
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Generated sample dataset with {size} URLs at {output_path}")
    
    return output_path

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print application header"""
    print("="*70)
    print("               PHISHING URL DETECTION SYSTEM")
    print("="*70)
    print("This system uses machine learning to identify potential phishing URLs")
    print("="*70)
    print()

def print_loading_animation(message, duration=3):
    """Display a simple loading animation"""
    characters = "|/-\\"
    for _ in range(duration * 10):
        for char in characters:
            sys.stdout.write(f"\r{message} {char}")
            sys.stdout.flush()
            time.sleep(0.025)
    print()

def display_risk_level(probability):
    """Display a visual risk level based on probability"""
    if probability < 0.2:
        risk = "Very Low"
        color_code = "\033[92m"  # Green
    elif probability < 0.4:
        risk = "Low"
        color_code = "\033[92m"  # Green
    elif probability < 0.6:
        risk = "Medium"
        color_code = "\033[93m"  # Yellow
    elif probability < 0.8:
        risk = "High"
        color_code = "\033[91m"  # Red
    else:
        risk = "Very High"
        color_code = "\033[91m"  # Red
    
    reset_code = "\033[0m"
    
    # For terminals that don't support color codes
    try:
        print(f"Risk Level: {color_code}{risk}{reset_code} ({probability:.2%})")
    except:
        print(f"Risk Level: {risk} ({probability:.2%})")
    
    # Visual bar
    try:
        bar_length = 40
        filled_length = int(probability * bar_length)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"{color_code}{bar}{reset_code}")
    except:
        bar_length = 40
        filled_length = int(probability * bar_length)
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        print(f"{bar}")

def display_feature_importance(features, top_n=5):
    """Display the most important features that contributed to the prediction"""
    # This is a simplified version - actual feature importance would come from the model
    suspicious_features = []
    
    # Check for common phishing indicators
    if features['suspicious_tld'] == 1:
        suspicious_features.append("Suspicious top-level domain (TLD)")
    if features['has_ip_address'] == 1:
        suspicious_features.append("Contains an IP address")
    if features['has_at_symbol'] == 1:
        suspicious_features.append("Contains @ symbol")
    if features['suspicious_words'] > 0:
        suspicious_features.append(f"Contains {features['suspicious_words']} suspicious words")
    if features['has_hex'] == 1:
        suspicious_features.append("Contains hexadecimal character codes")
    if features['url_length'] > 75:
        suspicious_features.append("Unusually long URL")
    if features['domain_in_path'] == 1:
        suspicious_features.append("Domain name appears in the path")
    if features['is_https'] == 0:
        suspicious_features.append("Not using HTTPS (secure connection)")
    
    print("\nSuspicious elements detected:")
    if suspicious_features:
        for i, feature in enumerate(suspicious_features[:top_n], 1):
            print(f"  {i}. {feature}")
    else:
        print("  None detected")

def main():
    # File paths
    sample_data_path = "phishing_dataset.csv"
    model_path = "phishing_detector_model.joblib"
    scaler_path = "phishing_detector_scaler.joblib"
    
    clear_screen()
    print_header()
    
    # Check if model exists, if not train a new one
    detector = PhishingURLDetector()
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print("First-time setup: A model needs to be trained.")
        print("This will generate sample data and train the detection model.")
        input("Press Enter to continue...")
        
        # Generate sample data and train model
        generate_sample_dataset(sample_data_path, size=2000)
        detector.train(sample_data_path)
        detector.save_model(model_path, scaler_path)
    else:
        print("Loading existing model...")
        print_loading_animation("Loading model")
        detector.load_model(model_path, scaler_path)
    
    clear_screen()
    print_header()
    
    # Main application loop
    while True:
        print("\nOptions:")
        print("1. Check a URL")
        print("2. Check multiple URLs")
        print("3. Retrain model")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            clear_screen()
            print_header()
            print("URL PHISHING CHECK")
            print("-" * 70)
            
            url = input("\nEnter the URL to check: ")
            if not url:
                print("No URL entered. Please try again.")
                continue
            
            # Add http:// if no protocol specified
            if not url.startswith('http'):
                url = 'http://' + url
            
            print_loading_animation("Analyzing URL")
            
            try:
                result = detector.predict(url)
                
                print("\nANALYSIS RESULTS")
                print("-" * 70)
                print(f"URL: {result['url']}")
                print(f"Verdict: {'⚠️  POTENTIAL PHISHING' if result['is_phishing'] else '✅  LIKELY LEGITIMATE'}")
                
                display_risk_level(result['probability'])
                display_feature_importance(result['features'])
                
                print("\nRECOMMENDATION:")
                if result['is_phishing']:
                    print("This URL shows characteristics common in phishing attempts.")
                    print("Exercise caution and do not enter sensitive information.")
                else:
                    print("This URL appears legitimate, but always stay vigilant online.")
            
            except Exception as e:
                print(f"\nError analyzing URL: {str(e)}")
                
            input("\nPress Enter to return to the main menu...")
            clear_screen()
            print_header()
                
        elif choice == '2':
            clear_screen()
            print_header()
            print("BULK URL CHECKING")
            print("-" * 70)
            print("Enter URLs (one per line). Enter a blank line when finished.")
            
            urls = []
            while True:
                url = input("> ")
                if not url:
                    break
                urls.append(url)
            
            if not urls:
                print("No URLs entered. Please try again.")
                continue
            
            print_loading_animation("Analyzing URLs")
            
            results = []
            for url in urls:
                # Add http:// if no protocol specified
                if not url.startswith('http'):
                    url = 'http://' + url
                
                try:
                    result = detector.predict(url)
                    results.append(result)
                except Exception as e:
                    print(f"\nError analyzing URL '{url}': {str(e)}")
            
            print("\nBULK ANALYSIS RESULTS")
            print("-" * 70)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['url']}")
                print(f"   Verdict: {'⚠️  POTENTIAL PHISHING' if result['is_phishing'] else '✅  LIKELY LEGITIMATE'}")
                print(f"   Risk: {result['probability']:.2%}")
            
            input("\nPress Enter to return to the main menu...")
            clear_screen()
            print_header()
                
        elif choice == '3':
            clear_screen()
            print_header()
            print("RETRAIN MODEL")
            print("-" * 70)
            
            confirm = input("This will generate a new dataset and retrain the model. Continue? (y/n): ")
            if confirm.lower() == 'y':
                print_loading_animation("Generating new dataset")
                generate_sample_dataset(sample_data_path, size=2000)
                
                print_loading_animation("Training new model")
                detector.train(sample_data_path)
                detector.save_model(model_path, scaler_path)
                
                print("\nModel retrained successfully!")
            
            input("\nPress Enter to return to the main menu...")
            clear_screen()
            print_header()
                
        elif choice == '4':
            print("\nThank you for using the Phishing URL Detection System. Goodbye!")
            break
            
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user. Goodbye!")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        print("The program will now exit.")
