package com.byld.plugin

import com.intellij.execution.configurations.GeneralCommandLine
import com.intellij.openapi.project.Project
import com.intellij.openapi.vfs.VirtualFile
import com.intellij.platform.lsp.api.LspServerSupportProvider
import com.intellij.platform.lsp.api.ProjectWideLspServerDescriptor
import java.io.File

class ByldLspServerSupportProvider : LspServerSupportProvider {
    override fun fileOpened(project: Project, file: VirtualFile, serverStarter: LspServerSupportProvider.LspServerStarter) {
        if (file.extension == "byd") {
            serverStarter.ensureServerStarted(ByldLspServerDescriptor(project))
        }
    }
}

class ByldLspServerDescriptor(project: Project) : ProjectWideLspServerDescriptor(project, "Byld") {
    override fun isSupportedFile(file: VirtualFile): Boolean {
        return file.extension == "byd"
    }

    override fun createCommandLine(): GeneralCommandLine {
        val baseDir = project.basePath
        if (baseDir != null) {
            val releasePath = File(baseDir, "target/release/byld-lsp")
            if (releasePath.exists()) {
                return GeneralCommandLine(releasePath.absolutePath)
            }
            val debugPath = File(baseDir, "target/debug/byld-lsp")
            if (debugPath.exists()) {
                return GeneralCommandLine(debugPath.absolutePath)
            }
        }
        return GeneralCommandLine("byld-lsp")
    }
}
