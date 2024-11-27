sample_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
times = {'Isolation Forest': [], 'IQR': [], 'Z-Score': []}

for size in sample_sizes:
    X_sample = X.sample(frac=size, random_state=42)

    # Measure time for Isolation Forest
    start_time = time.time()
    iso_forest.fit(X_sample)
    times['Isolation Forest'].append(time.time() - start_time)

    # Measure time for IQR
    start_time = time.time()
    Q1 = X_sample.quantile(0.25)
    Q3 = X_sample.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    (X_sample < lower_bound) | (X_sample > upper_bound)
    times['IQR'].append(time.time() - start_time)

    # Measure time for Z-Score
    start_time = time.time()
    z_scores = zscore(X_sample, nan_policy='omit')
    (np.abs(z_scores) > threshold)
    times['Z-Score'].append(time.time() - start_time)

# Display the results
print("Scalability results (time in seconds):")
print(pd.DataFrame(times, index=[f"{int(size * 100)}%" for size in sample_sizes]))
