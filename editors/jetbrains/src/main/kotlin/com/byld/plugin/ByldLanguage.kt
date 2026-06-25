package com.byld.plugin

import com.intellij.lang.Language

class ByldLanguage : Language("Byld") {
    companion object {
        val INSTANCE = ByldLanguage()
    }
}
