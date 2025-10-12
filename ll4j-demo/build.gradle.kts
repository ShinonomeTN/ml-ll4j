plugins {
    java
}

repositories {
    mavenCentral()
}

dependencies {
    implementation(project(":ll4j-huzpsb"))
    implementation(project(":ll4j-train"))
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(8)
    }
}