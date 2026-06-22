package com.byld.plugin

import com.intellij.openapi.editor.colors.TextAttributesKey
import com.intellij.openapi.fileTypes.SyntaxHighlighterBase
import com.intellij.openapi.fileTypes.SyntaxHighlighterFactory
import com.intellij.openapi.project.Project
import com.intellij.openapi.vfs.VirtualFile
import com.intellij.psi.tree.IElementType

class ByldSyntaxHighlighterFactory : SyntaxHighlighterFactory() {
    override fun getSyntaxHighlighter(project: Project?, virtualFile: VirtualFile?) = ByldSyntaxHighlighter()
}

class ByldSyntaxHighlighter : SyntaxHighlighterBase() {
    override fun getHighlightingLexer() = com.intellij.lexer.EmptyLexer()
    override fun getTokenHighlights(tokenType: IElementType?) = emptyArray<TextAttributesKey>()
}
