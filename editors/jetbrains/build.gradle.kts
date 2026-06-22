plugins {
    id("java")
    kotlin("jvm") version "1.9.20"
    id("org.jetbrains.intellij") version "1.16.1"
}

group = "com.byld"
version = "0.1.0"

repositories {
    mavenCentral()
}

intellij {
    version.set("2023.2.5")
    type.set("IC")
    plugins.set(listOf("java", "Kotlin"))
}

tasks {
    patchPluginXml {
        sinceBuild.set("232")
        untilBuild.set("242.*")
    }

    signPlugin {
        // Left empty for local builds
    }

    publishPlugin {
        // Left empty
    }
}
