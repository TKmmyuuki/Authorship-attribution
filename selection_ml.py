import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             make_scorer, RocCurveDisplay)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

def plot_feature_20importance(features_df):
    # Gráfico de barras das features
    plt.figure(figsize=(12, 8))
    plt.barh(features_df['feature'], features_df['mi_score'], color='skyblue')
    plt.xlabel('Mutual Information Score', fontsize=12)
    plt.title(f'Top 20 Features Based on Mutual Information', fontsize=14, pad=20)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(features_df, top_n=20, categories=None):
    if categories:
        # Analysis by categories
        category_importance = {}
        
        for category, keywords in categories.items():
            cat_features = [f for f in features_df['feature'] if any(k in f for k in keywords)]
            cat_importance = features_df[features_df['feature'].isin(cat_features)]['mi_score'].sum()
            
            category_importance[category] = {
                'importance': cat_importance,
                'count': len(cat_features),
                'features': cat_features,
                'avg_importance': cat_importance / len(cat_features) if cat_features else 0
            }
        
        # Bar chart by category
        plt.figure(figsize=(10, 6))
        categories_list = list(category_importance.keys())
        importance_values = [cat_info['importance'] for cat_info in category_importance.values()]
        
        colors = [
            '#ff9999',  # light red
            '#66b3ff',  # light blue
            '#99ff99',  # light green
            '#ffcc99',  # light orange
            '#c2c2f0',  # lavender
            '#ffb3e6',  # pink
            '#c2f0c2',  # mint green
            '#ff6666',  # stronger red
            '#66ffcc',  # turquoise
            '#ffd966'   # yellow
        ]
        
        plt.barh(categories_list, importance_values, color=colors[:len(categories_list)])
        plt.xlabel('Total Importance (Mutual Information)', fontsize=12)
        plt.ylabel('Feature Categories', fontsize=12)
        plt.title('Feature Importance by Category (Mutual Information)', fontsize=14, pad=15)
        plt.gca().invert_yaxis()  # mostra a categoria mais importante no topo
        plt.tight_layout()
        plt.show()

def validate_feature_selection(df, feature_names, target='origin', 
                             top_n_range=range(10, 51, 5), cv_folds=5,
                             correlation_threshold=0.8):
    # 1. Seleção inicial com MI
    print("📊 Calculando Mutual Information...")
    mi_df = feature_selection_chunked(df, feature_names, target, chunk_size=100)
    
    # 2. Remover features redundantes baseado em correlação
    print("🔍 Removendo features redundantes...")
    mi_df_non_redundant = remove_redundant_features(mi_df, df, correlation_threshold)
    
    # 3. Análise de correlação das top features após remoção de redundâncias
    print("📈 Analisando correlações após remoção de redundâncias...")
    top_features = mi_df_non_redundant.head(50)['feature'].tolist()
    plot_correlation_analysis(df, top_features, correlation_threshold)
    
    # 5. Análise final do melhor conjunto
    print("\n" + "="*60)
    print("📋 RELATÓRIO FINAL DA SELEÇÃO DE FEATURES")
    print("="*60)
    print(f"Total de features inicial: {len(feature_names)}")
    print(f"Features após MI: {len(mi_df)}")
    print(f"Features após remoção de redundâncias: {len(mi_df_non_redundant)}")
    
    return mi_df_non_redundant

def feature_selection_chunked(df, feature_names, target='origin', chunk_size=100):
    """
    Calcula Mutual Information em chunks e retorna DataFrame ordenado
    Retorna: DataFrame com colunas ['feature', 'mi_score']
    """
    from sklearn.feature_selection import mutual_info_classif
    import pandas as pd
    import numpy as np
    
    mi_scores = []
    
    # Processa em chunks para evitar memory errors
    for i in range(0, len(feature_names), chunk_size):
        chunk_features = feature_names[i:i + chunk_size]
        X_chunk = df[chunk_features].values
        y = df[target].values
        
        chunk_mi = mutual_info_classif(X_chunk, y, random_state=42)
        mi_scores.extend(zip(chunk_features, chunk_mi))
    
    # Cria DataFrame com os resultados
    mi_df = pd.DataFrame(mi_scores, columns=['feature', 'mi_score'])
    mi_df = mi_df.sort_values('mi_score', ascending=False).reset_index(drop=True)
    
    return mi_df

def remove_redundant_features(mi_df, df, correlation_threshold=0.8, additional_columns_to_remove=None):
    """
    Remove features redundantes baseado em correlação e colunas adicionais especificadas
    Retorna: DataFrame filtrado com colunas ['feature', 'mi_score']
    """
    # Colunas adicionais para remover
    default_columns_to_remove = ['ai', 'intelligence', 'artificial intelligence', 'artificial', 'education', 'students']
    if additional_columns_to_remove:
        default_columns_to_remove.extend(additional_columns_to_remove)
    
    features = mi_df['feature'].tolist()
    corr_matrix = df[features].corr().abs()
    
    features_to_keep = []
    features_to_remove = []
    
    # Primeiro, remove as colunas adicionais especificadas
    for column in default_columns_to_remove:
        if column in features:
            features_to_remove.append(column)
            print(f"   🗑️  Removendo '{column}' (coluna especificada)")
    
    # Itera pelas features ordenadas por importância (MI score)
    for i, feat1 in enumerate(features):
        if feat1 in features_to_remove:
            continue
            
        features_to_keep.append(feat1)
        
        # Verifica correlação com features menos importantes
        for j, feat2 in enumerate(features[i+1:], i+1):
            if (feat2 not in features_to_remove and 
                feat2 not in default_columns_to_remove and
                corr_matrix.iloc[i, j] > correlation_threshold):
                features_to_remove.append(feat2)
                print(f"   🔄 Removendo '{feat2}' (correlacionada com '{feat1}': {corr_matrix.iloc[i, j]:.3f})")
    
    print(f"📉 Total removido: {len(features_to_remove)} features (redundantes + especificadas)")
    print(f"📈 Total mantido: {len(features_to_keep)} features")
    
    # Filtra o DataFrame original mantendo as colunas originais
    filtered_df = mi_df[mi_df['feature'].isin(features_to_keep)].copy()
    return filtered_df.sort_values('mi_score', ascending=False)

def plot_correlation_analysis(df, features, threshold=0.8):
    """
    Analisa correlações entre as top features
    """
    if len(features) == 0:
        print("⚠️  Nenhuma feature para análise de correlação")
        return []
    
    corr_matrix = df[features].corr().abs()
    
    # Encontra pares altamente correlacionados
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                high_corr_pairs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    print(f"⚠️  {len(high_corr_pairs)} pares com correlação > {threshold}:")
    for feat1, feat2, corr in high_corr_pairs[:10]:  # Mostra apenas os 10 primeiros
        print(f"   {feat1} - {feat2}: {corr:.3f}")
    
    if len(high_corr_pairs) > 10:
        print(f"   ... e mais {len(high_corr_pairs) - 10} pares")
    
    # Heatmap das correlações
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='RdBu_r', center=0, 
                xticklabels=False, yticklabels=False)
    plt.title(f'Correlation Matrix Top {len(features)} Features')
    plt.tight_layout()
    plt.show()
    
    return high_corr_pairs


def random_forest_pipeline(X, y, test_size=0.2, random_state=42, n_estimators=100, cv=5):
    """
    Executa validação cruzada com StratifiedKFold para classificação binária balanceada.
    """
    
    # ==================== 2. VALIDAÇÃO CRUZADA COM ROC-AUC ROBUSTO ====================
    stratified_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Função personalizada para calcular ROC-AUC de forma segura
    def custom_cross_validate_with_auc(model, X, y, cv):
        auc_scores = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Clonar o modelo para cada fold
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            
            # Previsões
            y_pred = fold_model.predict(X_test)
            y_pred_proba = fold_model.predict_proba(X_test)[:, 1]
            
            # Calcular métricas
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
            recall_scores.append(recall_score(y_test, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
            
            # Calcular ROC-AUC apenas se houver ambas as classes no teste
            if len(np.unique(y_test)) == 2:
                auc_scores.append(roc_auc_score(y_test, y_pred_proba))
            else:
                auc_scores.append(np.nan)
        
        return {
            'test_accuracy': np.array(accuracy_scores),
            'test_precision': np.array(precision_scores),
            'test_recall': np.array(recall_scores),
            'test_f1': np.array(f1_scores),
            'test_roc_auc': np.array(auc_scores)
        }
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Executar validação cruzada personalizada
    results = custom_cross_validate_with_auc(clf, X, y, stratified_cv)
    
    print("\n📊 RESULTADOS DA VALIDAÇÃO CRUZADA:")
    print("-" * 40)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        scores = results['test_' + metric]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{metric:12}: {mean_score:.4f} ± {std_score:.4f}")
    
    # Resultados do ROC-AUC
    auc_scores = results['test_roc_auc']
    valid_auc_scores = auc_scores[~np.isnan(auc_scores)]
    
    if len(valid_auc_scores) > 0:
        mean_auc = np.mean(valid_auc_scores)
        std_auc = np.std(valid_auc_scores)
        print(f"roc_auc     : {mean_auc:.4f} ± {std_auc:.4f} "
              f"(baseado em {len(valid_auc_scores)}/{cv} folds válidos)")
    else:
        print("roc_auc     : NaN (nenhum fold válido para cálculo)")
    
    # ==================== 3. TREINAMENTO FINAL ====================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y, shuffle=True
    )

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    # Métricas finais
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n📈 MÉTRICAS NO CONJUNTO DE TESTE:")
    print("-" * 30)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # ==================== 4. MATRIZ DE CONFUSÃO ====================
    cm = confusion_matrix(y_test, y_pred)
    totals = cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Matriz de Confusão
    im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    ax1.set_title('Confusion Matrix - Random Forest\n(Test Set)', fontsize=14, pad=20)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Human', 'IA'])
    ax1.set_yticklabels(['Human', 'IA'])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, f"{cm[i, j]}/{totals[i,0]}",
                    ha='center', va='center', color='black', fontsize=12)

    # ==================== 5. CURVA ROC ====================
    RocCurveDisplay.from_estimator(rf_model, X_test, y_test, ax=ax2)
    ax2.set_title(f'Curva ROC (AUC = {roc_auc:.4f})')
    ax2.plot([0, 1], [0, 1], 'k--', label='Classificador Aleatório (AUC = 0.5)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return rf_model, cm, results


def logistic_regression_pipeline(X, y, test_size=0.2, random_state=42, cv=5, max_iter=1000):
    """
    Executa validação cruzada com StratifiedKFold para Logistic Regression
    com pré-processamento adequado baseado nas distribuições das features.
    """
    
    # ==================== 1. VERIFICAÇÃO E INFORMAÇÕES ====================
    unique_classes = np.unique(y)
    if len(unique_classes) != 2:
        raise ValueError("❌ Esta função é apenas para classificação binária.")
    
    print(f"🔍 Classificação Binária: Classes {unique_classes}")
    print(f"📊 Distribuição:")
    print(f"   Classe {unique_classes[0]}: {np.sum(y == unique_classes[0])} amostras")
    print(f"   Classe {unique_classes[1]}: {np.sum(y == unique_classes[1])} amostras")
    
    # ==================== 2. DEFINIÇÃO DO PRÉ-PROCESSAMENTO ====================
    # Identificar colunas para cada tipo de scaler baseado nas distribuições
    if hasattr(X, 'columns'):
        features_standard = [
            'lexical_word_count', 'lexical_avg_word_length', 
            'lexical_unique_words', 'syntactic_post_bigram_entropy'
        ]
        
        features_minmax = [
            'structural_hashtag_density', 'structural_extra_spaces',
            'stylistic_exclamation_density', 'stylistic_emoji_density',
            'stylistic_repeated_chars', 'syntactic_question_density',
            'syntactic_punct_ratio', 'structural_has_hashtag',
            'structural_has_mention', 'lexical_word_length_variance',
            'lexical_type_token_ratio', 'syntactic_pos_tag_entropy',
            'syntactic_comma_ratio', 'lexical_stopword_ratio'
        ]
        
        # Filtrar apenas as colunas que existem no DataFrame
        features_standard = [col for col in features_standard if col in X.columns]
        features_minmax = [col for col in features_minmax if col in X.columns]
        
        print(f"🔧 Pré-processamento:")
        print(f"   StandardScaler: {len(features_standard)} features")
        print(f"   MinMaxScaler: {len(features_minmax)} features")
        
        # Criar o ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('standard', StandardScaler(), features_standard),
                ('minmax', MinMaxScaler(), features_minmax)
            ],
            remainder='passthrough'
        )
    else:
        # Se não for DataFrame, usar StandardScaler em tudo
        print("⚠️  X não é DataFrame, usando StandardScaler em todas as features")
        preprocessor = StandardScaler()
    
    # ==================== 3. PIPELINE E VALIDAÇÃO CRUZADA ====================
    stratified_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Criar pipeline com pré-processamento e modelo
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            class_weight='balanced'  # Importante para lidar com possíveis desbalanceamentos
        ))
    ])
    
    # Função personalizada para validação cruzada
    def custom_cross_validate_with_auc(pipeline, X, y, cv):
        auc_scores = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx], \
                             X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
            y_train, y_test = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx], \
                             y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx]
            
            # Clonar o pipeline para cada fold
            fold_pipeline = clone(pipeline)
            fold_pipeline.fit(X_train, y_train)
            
            # Previsões
            y_pred = fold_pipeline.predict(X_test)
            y_pred_proba = fold_pipeline.predict_proba(X_test)[:, 1]
            
            # Calcular métricas
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
            recall_scores.append(recall_score(y_test, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
            
            # Calcular ROC-AUC
            if len(np.unique(y_test)) == 2:
                auc_scores.append(roc_auc_score(y_test, y_pred_proba))
            else:
                auc_scores.append(np.nan)
        
        return {
            'test_accuracy': np.array(accuracy_scores),
            'test_precision': np.array(precision_scores),
            'test_recall': np.array(recall_scores),
            'test_f1': np.array(f1_scores),
            'test_roc_auc': np.array(auc_scores)
        }
    
    # Executar validação cruzada personalizada
    results = custom_cross_validate_with_auc(pipeline, X, y, stratified_cv)
    
    print("\n📊 RESULTADOS DA VALIDAÇÃO CRUZADA (Logistic Regression):")
    print("-" * 60)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        scores = results['test_' + metric]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{metric:12}: {mean_score:.4f} ± {std_score:.4f}")
    
    # Resultados do ROC-AUC
    auc_scores = results['test_roc_auc']
    valid_auc_scores = auc_scores[~np.isnan(auc_scores)]
    
    if len(valid_auc_scores) > 0:
        mean_auc = np.mean(valid_auc_scores)
        std_auc = np.std(valid_auc_scores)
        print(f"roc_auc     : {mean_auc:.4f} ± {std_auc:.4f} "
              f"(baseado em {len(valid_auc_scores)}/{cv} folds válidos)")
    else:
        print("roc_auc     : NaN (nenhum fold válido para cálculo)")
    
    # ==================== 4. TREINAMENTO FINAL ====================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y, shuffle=True
    )

    # Treinar o pipeline final
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            class_weight='balanced'
        ))
    ])
    
    final_pipeline.fit(X_train, y_train)

    y_pred = final_pipeline.predict(X_test)
    y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]

    # Métricas finais
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n📈 MÉTRICAS NO CONJUNTO DE TESTE:")
    print("-" * 30)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # ==================== 5. MATRIZ DE CONFUSÃO ====================
    cm = confusion_matrix(y_test, y_pred)
    totals = cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Matriz de Confusão
    im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    ax1.set_title('Confusion Matrix - Logistic Regression\n(Test Set)', fontsize=14, pad=20)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Humano', 'IA'])
    ax1.set_yticklabels(['Humano', 'IA'])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, f"{cm[i, j]}/{totals[i,0]}",
                    ha='center', va='center', color='black', fontsize=12)

    # ==================== 6. CURVA ROC ====================
    RocCurveDisplay.from_estimator(final_pipeline, X_test, y_test, ax=ax2)
    ax2.set_title(f'ROC Curve (AUC = {roc_auc:.4f})')
    ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ==================== 7. COEFICIENTES DA REGRESSÃO ====================
    if hasattr(X, 'columns'):
        # Extrair os coeficientes do modelo
        classifier = final_pipeline.named_steps['classifier']
        
        # Obter os nomes das features após o pré-processamento
        try:
            feature_names = []
            for name, transformer, features in preprocessor.transformers:
                if name != 'remainder':
                    feature_names.extend(features)
            
            coefficients = classifier.coef_[0]
            
            # Ordenar features por importância absoluta
            indices = np.argsort(np.abs(coefficients))[::-1]
            
            print(f"\n🎯 TOP 15 COEFICIENTES MAIS IMPORTANTES:")
            print("-" * 40)
            for i, idx in enumerate(indices[:15]):
                sign = "+" if coefficients[idx] > 0 else "-"
                print(f"{i+1:2}. {feature_names[idx]:25}: {sign} {abs(coefficients[idx]):.4f}")
            
            # Plot dos coeficientes
            plt.figure(figsize=(12, 8))
            top_features = min(15, len(feature_names))
            colors = ['red' if coef < 0 else 'blue' for coef in coefficients[indices[:top_features]]]
            plt.barh(range(top_features), coefficients[indices[:top_features]][::-1], color=colors[::-1])
            plt.yticks(range(top_features), [feature_names[i] for i in indices[:top_features]][::-1])
            plt.xlabel('Valor do Coeficiente')
            plt.title('Top 15 Features Mais Importantes (Logistic Regression)')
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.8)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"⚠️  Não foi possível extrair coeficientes: {e}")

    return final_pipeline, cm, results



def svm_pipeline(X, y, test_size=0.2, random_state=42, cv=5, kernel='linear'):
    """
    Pipeline corrigido para SVM com CalibratedClassifierCV para evitar NaN no ROC-AUC
    """
    
    # ==================== 1. CONFIGURAÇÃO INICIAL ====================
    unique_classes = np.unique(y)
    print(f"🔍 Classes: {unique_classes}")
    print(f"📊 Distribuição: {np.bincount(y)}")
    
    # ==================== 2. PRÉ-PROCESSAMENTO ====================
    if hasattr(X, 'columns'):
        preprocessor = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), X.columns)
            ],
            remainder='passthrough'
        )
    else:
        preprocessor = StandardScaler()
    
    # ==================== 3. MODELO COM CALIBRAÇÃO ====================
    # Usar SVM sem probabilidades e calibrar depois
    base_svm = SVC(
        kernel=kernel, 
        random_state=random_state, 
        class_weight='balanced',
        probability=False  # Desativar probabilidades nativas do SVM
    )
    
    # Usar CalibratedClassifierCV para probabilidades mais confiáveis
    calibrated_svm = CalibratedClassifierCV(
        base_svm, 
        cv=3, 
        method='sigmoid'
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', calibrated_svm)
    ])
    
    # ==================== 4. VALIDAÇÃO CRUZADA ====================
    stratified_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Métricas para validação cruzada
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall', 
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    print("🔄 Executando validação cruzada...")
    cv_results = cross_validate(
        pipeline, X, y, 
        cv=stratified_cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
        error_score='raise'
    )
    
    print("\n📊 RESULTADOS DA VALIDAÇÃO CRUZADA (SVM):")
    print("-" * 60)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    for metric in metrics:
        scores = cv_results[f'test_{metric}']
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{metric:12}: {mean_score:.4f} ± {std_score:.4f}")
    
    # ==================== 5. TREINAMENTO FINAL ====================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y, shuffle=True
    )

    print("🔧 Treinando modelo final...")
    pipeline.fit(X_train, y_train)

    # Previsões
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    
    # Métricas finais
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

    print(f"\n📈 MÉTRICAS NO CONJUNTO DE TESTE:")
    print("-" * 30)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # ==================== 6. MATRIZ DE CONFUSÃO ====================
    cm = confusion_matrix(y_test, y_pred)
    totals = cm.sum(axis=1)[:, np.newaxis]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Matriz de Confusão
    im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    ax1.set_title('Confusion Matrix - SVM\n(Test Set)', fontsize=14, pad=20)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Human', 'IA'])
    ax1.set_yticklabels(['Human', 'IA'])

    # Adicionar valores na matriz 
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, f"{cm[i, j]}/{totals[i,0]}",
                    ha='center', va='center', color='black', fontsize=12)

    # ==================== 7. CURVA ROC ====================
    try:
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax2)
        ax2.set_title(f'ROC Curve (AUC = {roc_auc:.4f})')
        ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    except Exception as e:
        ax2.text(0.5, 0.5, f'Erro na curva ROC:\n{str(e)}', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=10)
        ax2.set_title('Erro na Curva ROC')

    plt.tight_layout()
    plt.show()
    
    return pipeline, cm, cv_results

