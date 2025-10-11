plugins {
    `java`
    `application`
}

repositories {
    mavenCentral()
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(8)
    }
}

application {
    mainClass.set("huzpsb.ll4j.samples.TestMinRt")
}