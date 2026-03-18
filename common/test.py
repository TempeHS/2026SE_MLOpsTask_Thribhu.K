import sys, os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from utils import SentimentAnalyser
import numpy as np

analyser = SentimentAnalyser("Model_Development")

print("Model classes:", analyser.model.classes_)
print("Coef shape:", analyser.model.coef_.shape)

# Top positive contributors (class 1)
tfidf_names = np.array(analyser.tfidf.get_feature_names_out())
coef = analyser.model.coef_[0]

top_pos = np.argsort(coef)[-10:][::-1]
top_neg = np.argsort(coef)[:10]

print("\nTop words pushing toward class 1 (should be one sentiment):")
for i in top_pos:
    print(f"  {tfidf_names[i]:<15} {coef[i]:.4f}")

print("\nTop words pushing toward class 0:")
for i in top_neg:
    print(f"  {tfidf_names[i]:<15} {coef[i]:.4f}")
