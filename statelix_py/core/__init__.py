try:
    from .statelix_core import (
        # General
        OLSResult, fit_ols_full, predict_ols,
        
        # Graph
        graph,
        
        # Causal
        causal,
        
        # Bayes
        bayes,
        hmc_sample,
        
        # Search
        search,
        
        # Other modules
        KMeansResult, fit_kmeans,
        AnoVaResult, f_oneway,
        ARResult, fit_ar,
        LogisticRegression, LogisticResult,
        DTW,
        KDTree,
        ChangePointDetector,
        KalmanFilter,
        GradientBoostingRegressor,
        FactorizationMachine, FMTask,
        SparseMatrix
    )
except ImportError:
    # Allow importing package during build/install without extension present
    pass
