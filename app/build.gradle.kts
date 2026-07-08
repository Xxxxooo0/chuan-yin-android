plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.gvcrt.clean"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.gvcrt.clean"
        minSdk = 27
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
            }
        }
    }

    sourceSets {
        getByName("main") {
            if (project.hasProperty("gvcrtSkipAssets")) {
                assets.setSrcDirs(emptyList<String>())
            } else {
                assets.srcDirs("src/main/assets")
            }
        }
    }

    androidResources {
        noCompress += listOf("tflite", "onnx", "mnn", "bin", "f32le", "gvc", "dla")
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    implementation(files("../mtk/Android_V_neuropilot_240408.aar"))
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")
    implementation("org.tensorflow:tensorflow-lite:2.17.0")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.17.0")
    implementation("org.tensorflow:tensorflow-lite-gpu-api:2.17.0")
}
