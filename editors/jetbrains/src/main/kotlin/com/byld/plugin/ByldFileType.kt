package com.byld.plugin

import com.intellij.openapi.fileTypes.LanguageFileType
import javax.swing.Icon

class ByldFileType : LanguageFileType(ByldLanguage.INSTANCE) {
    companion object {
        @JvmField
        val INSTANCE = ByldFileType()
    }

    override fun getName() = "Byld File"
    override fun getDescription() = "Byld UI layout file"
    override fun getDefaultExtension() = "byd"
    override fun getIcon(): Icon? = null
}
