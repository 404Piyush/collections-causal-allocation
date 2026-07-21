# References

Academic and technical references underpinning the methods in this project.

## Regression Discontinuity Design

1. **Hahn, J., Todd, P., & Van der Klaauw, W.** (2001).
   *Identification and Estimation of Treatment Effects with a
   Regression Discontinuity Design.* **Econometrica**, 69(1), 201-209.
   — Foundational paper establishing the local randomization
   interpretation of sharp RDD.

2. **Imbens, G. W., & Kalyanaraman, K.** (2012).
   *Optimal Bandwidth Choice for the Regression Discontinuity
   Estimator.* **Review of Economic Studies**, 79(3), 933-959.
   — MSE-optimal bandwidth selector used in `src/rdd/bandwidth.py`.

3. **McCrary, J.** (2008).
   *Manipulation of the Running Variable in the Regression Discontinuity
   Design: A Density Test.* **Journal of Econometrics**, 142(2), 698-714.
   — Density / manipulation test used in `src/rdd/density_test.py`.

4. **Lee, D. S., & Lemieux, T.** (2010).
   *Regression Discontinuity Designs in Economics.* **Journal of Economic
   Literature**, 48(2), 281-355.
   — Comprehensive review of RDD practice.

5. **Korting, C., Lieberman, C., & Shrader, J.** (2021, working paper).
   *RDD with a Mismeasured Running Variable.*
   — Motivates the covariate-balance check in `src/rdd/covariate_balance.py`.

## Local Linear Regression

6. **Fan, J., & Gijbels, I.** (1996).
   *Local Polynomial Modelling and Its Applications.* **Chapman & Hall**.
   — Theoretical basis for local-linear regression with a triangular
   kernel.

7. **Wand, M. P., & Jones, M. C.** (1995).
   *Kernel Smoothing.* **Chapman & Hall / CRC**.
   — Standard reference on kernel methods, including the triangular
   kernel used here.

## Mixed-Integer Linear Programming

8. **Wolsey, L. A.** (1998).
   *Integer Programming.* **Wiley**.
   — Standard reference on MILP modeling and branch-and-bound.

9. **Mitchell, S., OSullivan, M., & Dunning, I.** (2011).
   *PuLP: A Linear Programming Toolkit for Python.*
   — Reference for the `PuLP` modeling library used in `src/milp/`.

10. **Forrest, J., & Lougee-Heimer, R.** (2005).
    *CBC User Guide.* **COIN-OR**.
    — Reference for the `COIN-OR Branch and Cut` solver used as the
    default MILP backend.

## Operations Research / Collections Optimization

11. **Berger, A. N., & Udell, G. F.** (1990).
    *Collateral, Loan Quality, and Bank Risk.* **Journal of Monetary
    Economics**, 25(1), 21-42.
    — Background on lender risk models.

12. **Canner, G. B., & Duffin, M. L.** (1995).
    *Collections Scoring: The State of the Art.* **Fed Reserve Bulletin**.
    — Early work on scoring delinquent accounts.

## Software

13. **Harris, C. R., et al.** (2020).
    *Array programming with NumPy.* **Nature**, 585(7825), 357-362.

14. **McKinney, W.** (2010).
    *Data Structures for Statistical Computing in Python.*
    **Proceedings of the 9th Python in Science Conference**, 56-61.

15. **Hunter, J. D.** (2007).
    *Matplotlib: A 2D Graphics Environment.* **Computing in Science &
    Engineering**, 9(3), 90-95.

## How to cite this project

See [`CITATION.cff`](../CITATION.cff) for machine-readable citation
metadata, or click "Cite this repository" on the GitHub page.