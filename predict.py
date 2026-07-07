import numpy as np


# =========================
# THRESHOLDS
# =========================
T_A = 0.301      # Model A threshold
T_A1 = 0.50      # change après calibration A1


# =========================
# FINAL DECISION
# =========================
def final_decision(pA, pA1, probs_B3):
    """
    pA       = prob abnormal from Model A
    pA1      = prob abnormal from Model A1
    probs_B3 = [p_MI, p_STTC, p_CD]
    """

    # Step 1: A + A1 check normal
    if pA < T_A and pA1 < T_A1:
        return "NORMAL"

    # Step 2: B3 chooses disease
    classes = ["MI", "STTC", "CD"]
    return classes[int(np.argmax(probs_B3))]


# =========================
# TEST EXAMPLES
# =========================
if __name__ == "__main__":

    # Example 1: normal
    pA = 0.10
    pA1 = 0.20
    probs_B3 = [0.20, 0.60, 0.20]

    print("Example 1:", final_decision(pA, pA1, probs_B3))

    # Example 2: abnormal → CD
    pA = 0.70
    pA1 = 0.80
    probs_B3 = [0.15, 0.25, 0.60]

    print("Example 2:", final_decision(pA, pA1, probs_B3))

    # Example 3: abnormal → MI
    pA = 0.80
    pA1 = 0.75
    probs_B3 = [0.65, 0.25, 0.10]

    print("Example 3:", final_decision(pA, pA1, probs_B3))