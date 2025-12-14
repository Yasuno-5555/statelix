cmake_minimum_required(VERSION 3.18)
project(statelix_core VERSION 2.2.0 LANGUAGES CXX)

# C++17を要求
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ビルドタイプのデフォルト
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# コンパイラフラグ
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# 依存関係の検索
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(pybind11 CONFIG)

# pybind11が見つからない場合は、Pythonから取得
if(NOT pybind11_FOUND)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    find_package(pybind11 CONFIG REQUIRED)
endif()

# ソースファイルの収集
set(CORE_SOURCES
    src/linear_model/ols.cpp
    src/cluster/kmeans.cpp
    src/stats/anova.cpp
    src/time_series/ar_model.cpp
    src/glm/glm_models.cpp
    src/survival/cox.cpp
    src/linear_model/elastic_net.cpp
    src/time_series/dtw.cpp
    src/search/kdtree.cpp
)

set(CORE_HEADERS
    src/linear_model/ols.h
    src/cluster/kmeans.h
    src/stats/anova.h
    src/time_series/ar_model.h
    src/glm/glm_models.h
    src/survival/cox.h
    src/linear_model/ridge.h
    src/linear_model/elastic_net.h
    src/optimization/optimization.h
    src/time_series/dtw.h
    src/search/kdtree.h
)

# Python モジュールの作成
pybind11_add_module(statelix_core MODULE
    src/bindings/python_bindings.cpp
    ${CORE_SOURCES}
)

# インクルードディレクトリ
target_include_directories(statelix_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ライブラリのリンク
target_link_libraries(statelix_core PRIVATE
    Eigen3::Eigen
)

# コンパイラ警告
if(MSVC)
    target_compile_options(statelix_core PRIVATE /W4)
else()
    target_compile_options(statelix_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# インストール設定
install(TARGETS statelix_core
    LIBRARY DESTINATION statelix_py
    RUNTIME DESTINATION statelix_py
)

# テストの有効化
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    
    # Google Test を探す（オプション）
    find_package(GTest)
    if(GTest_FOUND)
        add_subdirectory(tests/cpp)
    else()
        message(STATUS "Google Test not found, C++ tests will be skipped")
    endif()
endif()

# スタンドアロン実行ファイル（オプション）
option(BUILD_STANDALONE "Build standalone executable" OFF)
if(BUILD_STANDALONE)
    add_executable(statelix_standalone src/main.cpp ${CORE_SOURCES})
    target_include_directories(statelix_standalone PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(statelix_standalone PRIVATE Eigen3::Eigen)
endif()

# 情報表示
message(STATUS "Statelix v${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
message(STATUS "pybind11 version: ${pybind11_VERSION}")
