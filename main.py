from utils.imports import *
from utils.model_utils import save_model

def map_fault_type(fault_type):
    fault_names = {
        0: 'Normal',
        1: 'CF',
        2: 'EO',
        3: 'FWC',
        4: 'FWE',
        5: 'NC',
        6: 'RL'
    }
    return fault_names.get(fault_type, 'Unknown')

def add_value_labels(ax, spacing=5):
    """Add labels to the bars in a bar chart."""
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        space = spacing
        va = 'bottom'
        label = f"{y_value:.1%}"
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(0, space),
            textcoords="offset points",
            ha='center',
            va=va
        )

def process_file(csv_file, folder_name, log_dir):
    data = pd.read_csv(csv_file)
    data = data.drop(columns=['TWE_set'], errors='ignore')
    X = data.drop('CATEGORY', axis=1)
    y = data['CATEGORY']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Feature selection using MI and RFE
    mi_scores = mutual_info_classif(X_scaled, y)
    mi_selected_features = np.argsort(mi_scores)[-20:]

    rf_for_rfe = RandomForestClassifier(random_state=42)
    rfe = RFE(rf_for_rfe, n_features_to_select=10)
    rfe.fit(X_scaled.iloc[:, mi_selected_features], y)
    rfe_selected_features = X_scaled.columns[mi_selected_features][rfe.support_]

    X_selected = X_scaled[rfe_selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Run experiment with all models and ensemble methods
    experiment_name = f"{folder_name}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    results = run_experiment(X_train, X_test, y_train, y_test, experiment_name, log_dir, folder_name)

    # Visualization
    visualize_results(results, X_test, y_test, folder_name, log_dir)

def run_experiment(X_train, X_test, y_train, y_test, experiment_name, log_dir, folder_name):
    results = []

    # Random Forest
    print(f"Running Random Forest for {experiment_name}...")
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, n_jobs=-1)
    rf_grid_search.fit(X_train, y_train)
    best_rf_classifier = rf_grid_search.best_estimator_
    rf_accuracy = accuracy_score(y_test, best_rf_classifier.predict(X_test))
    results.append(('Random Forest', rf_accuracy))

    # SVM
    print(f"Running SVM for {experiment_name}...")
    svm_classifier = SVC(random_state=42)
    svm_classifier.fit(X_train, y_train)
    svm_accuracy = accuracy_score(y_test, svm_classifier.predict(X_test))
    results.append(('SVM', svm_accuracy))

    # Gaussian Naive Bayes
    print(f"Running GaussianNB for {experiment_name}...")
    gn_classifier = GaussianNB()
    gn_classifier.fit(X_train, y_train)
    gn_accuracy = accuracy_score(y_test, gn_classifier.predict(X_test))
    results.append(('GaussianNB', gn_accuracy))

    # CNN Model
    print(f"Running CNN for {experiment_name}...")
    X_train_cnn = np.expand_dims(X_train, axis=2)
    X_test_cnn = np.expand_dims(X_test, axis=2)
    y_train_cnn = to_categorical(y_train)
    y_test_cnn = to_categorical(y_test)

    cnn_model = Sequential()
    cnn_model.add(Input(shape=(X_train_cnn.shape[1], 1)))
    cnn_model.add(Conv1D(64, 2, activation='relu'))
    cnn_model.add(Conv1D(64, 2, activation='relu'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128, activation='relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(y_train_cnn.shape[1], activation='softmax'))

    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    log_file = os.path.join(log_dir, f"{experiment_name}_cnn_training_log.csv")
    with open(log_file, 'w') as f:
        f.write("Epoch,Loss,Accuracy,Val_Loss,Val_Accuracy\n")

    class LossHistoryCallback(tf.keras.callbacks.Callback):
        def __init__(self, log_file):
            super(LossHistoryCallback, self).__init__()
            self.log_file = log_file

        def on_epoch_end(self, epoch, logs=None):
            with open(self.log_file, 'a') as f:
                f.write(f"{epoch + 1},{logs['loss']},{logs['accuracy']},{logs['val_loss']},{logs['val_accuracy']}\n")

    cnn_model.fit(X_train_cnn, y_train_cnn, epochs=10, batch_size=32, verbose=1, 
                  validation_data=(X_test_cnn, y_test_cnn), callbacks=[LossHistoryCallback(log_file)])
    
    cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test_cnn, verbose=0)[1]
    results.append(('CNN', cnn_accuracy))

    # BN-RFEMI with stacking
    print(f"Running BN-RFEMI for {experiment_name}...")
    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('svc', SVC(kernel='linear', probability=True, random_state=42))
    ]

    rfemi_classifier = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(),
        cv=5
    )

    rfemi_classifier.fit(X_train, y_train)
    rfemi_accuracy = accuracy_score(y_test, rfemi_classifier.predict(X_test))
    results.append(('BN-RFEMI (Stacking)', rfemi_accuracy))

    # Voting Classifier
    print(f"Running Voting Classifier for {experiment_name}...")
    voting_classifier = VotingClassifier(
        estimators=base_learners,
        voting='soft'  # 'hard' for majority voting, 'soft' for averaging probabilities
    )
    voting_classifier.fit(X_train, y_train)
    voting_accuracy = accuracy_score(y_test, voting_classifier.predict(X_test))
    results.append(('Voting Classifier', voting_accuracy))

    # Bagging Classifier
    print(f"Running Bagging Classifier for {experiment_name}...")
    bagging_classifier = BaggingClassifier(
        base_estimator=RandomForestClassifier(random_state=42),
        n_estimators=10, random_state=42
    )
    bagging_classifier.fit(X_train, y_train)
    bagging_accuracy = accuracy_score(y_test, bagging_classifier.predict(X_test))
    results.append(('Bagging Classifier', bagging_accuracy))

    # Gradient Boosting Classifier
    print(f"Running Gradient Boosting for {experiment_name}...")
    gradient_boosting_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gradient_boosting_classifier.fit(X_train, y_train)
    gb_accuracy = accuracy_score(y_test, gradient_boosting_classifier.predict(X_test))
    results.append(('Gradient Boosting', gb_accuracy))

    # Logging results to CSV
    results_file = os.path.join(log_dir, f"{experiment_name}_results.csv")
    with open(results_file, 'w') as f:
        f.write("Model,Accuracy\n")
        for model, accuracy in results:
            f.write(f"{model},{accuracy:.4f}\n")

    # saving models
    rf_classifier = best_rf_classifier.fit(X_train, y_train)
    save_model(rf_classifier, "RandomForest", folder_name)
    
    #SVM 
    svm_classifier = SVC(random_state=42).fit(X_train, y_train)
    save_model(svm_classifier, "SVM", folder_name)
    
    # CNN
    cnn_model.save(os.path.join(log_dir,f"{folder_name}_CNN_MODEL.h5"))
    # print(f"CNN Model saved to: {os.path.join(log_dir,f"{folder_name}_CNN_MODEL.h5")}")
    print(f"CNN Model saved to log dir")
    
    return results

def visualize_results(results, X_test, y_test, folder_name, log_dir):
    models = [result[0] for result in results]
    accuracies = [result[1] for result in results]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Overall Accuracy Comparison
    bars = axes[0].bar(models, accuracies, color=['blue', 'orange', 'green', 'purple', 'red', 'cyan', 'magenta', 'yellow'])
    axes[0].set_title('Overall Accuracy Comparison', pad=20)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xticklabels(models, rotation=45)
    add_value_labels(axes[0])

    fault_types = list(range(7))
    fault_labels = [map_fault_type(f) for f in fault_types]
    
    # Calculate fault type accuracies for each model
    bar_width = 0.15
    index = np.arange(len(fault_types))

    for i, (model_name, _) in enumerate(results):
        if model_name == 'Random Forest':
            model = RandomForestClassifier().fit(X_test, y_test)
        elif model_name == 'SVM':
            model = SVC().fit(X_test, y_test)
        elif model_name == 'GaussianNB':
            model = GaussianNB().fit(X_test, y_test)
        elif model_name == 'CNN':
            # Add CNN-specific preparation if needed
            continue
        elif model_name == 'BN-RFEMI (Stacking)':
            model = StackingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
                    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
                    ('svc', SVC(kernel='linear', probability=True, random_state=42))
                ],
                final_estimator=LogisticRegression(),
                cv=5
            ).fit(X_test, y_test)
        elif model_name == 'Voting Classifier':
            model = VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
                    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
                    ('svc', SVC(kernel='linear', probability=True, random_state=42))
                ],
                voting='soft'
            ).fit(X_test, y_test)
        elif model_name == 'Bagging Classifier':
            model = BaggingClassifier(
                base_estimator=RandomForestClassifier(random_state=42),
                n_estimators=10, random_state=42
            ).fit(X_test, y_test)
        elif model_name == 'Gradient Boosting':
            model = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_test, y_test)
        else:
            continue

        # Compute fault type accuracies
        model_fault_accuracies = confusion_matrix(y_test, model.predict(X_test), normalize='true').diagonal()

        # Plot the fault type accuracies
        axes[1].bar(index + i * bar_width, model_fault_accuracies, bar_width, label=model_name)

    axes[1].set_title('Accuracy by Fault Type', pad=20)
    axes[1].set_xticks(index + 2 * bar_width)
    axes[1].set_xticklabels(fault_labels, rotation=45)
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Create a table beside the graph with accuracy percentages
    table_data = {
        'Fault Type': fault_labels,
    }
    for i, (model_name, _) in enumerate(results):
        model_fault_accuracies = confusion_matrix(y_test, model.predict(X_test), normalize='true').diagonal()
        table_data[f'{model_name} Accuracy'] = model_fault_accuracies * 100

    table_df = pd.DataFrame(table_data)
    axes[2].axis('off')
    axes[2].table(cellText=table_df.values, colLabels=table_df.columns, loc='center')

    # Placeholder for other visualizations, e.g., ground truth vs predictions
    axes[3].axis('off')  # If no additional visualizations, keep this axis empty

    plt.tight_layout()

    # Save the figure with a new name
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_filename = os.path.join(log_dir, f"{folder_name}_{timestamp}_comparison_performance.png")
    plt.savefig(output_filename)
    plt.close()

def process_folder(folder_path, folder_name, log_dir):
    for file_name in os.listdir(folder_path):
        if file_name.startswith('Copy of level') and file_name.endswith('.csv'):
            csv_file = os.path.join(folder_path, file_name)
            process_file(csv_file, folder_name, log_dir)

def main():
    base_dir = "data"
    severity_folders = ['SL-1', 'SL-2', 'SL-3', 'SL-4']
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    for folder_name in severity_folders:
        folder_path = os.path.join(base_dir, folder_name)
        process_folder(folder_path, folder_name, log_dir)

if __name__ == "__main__":
    main()