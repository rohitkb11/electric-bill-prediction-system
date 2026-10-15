# Implementation phases

1. **Data and preprocessing:** Keep one household CSV; select household-profile features and the monthly kWh target; clean and split before fitting preprocessing.
2. **Linear Regression:** Train the baseline, run five-fold parameter search, evaluate on held-out households, export the complete pipeline and generate Matplotlib plots.
3. **Application:** Express validates household details, calls Python and calculates a monthly BDT bill. The plain frontend displays the household form, prediction and evaluation.
4. **Verification and documentation:** Test the model artifact and Express integration; verify the form and graphs; document all seven PR12 stages and publish to GitHub.

The project requires no database, login, frontend framework, previous readings or additional model algorithms.
