import * as path from 'path';
import * as fs from 'fs';
import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    Executable
} from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('byld');
    let serverPath = config.get<string | null>('lsp.serverPath');

    if (!serverPath) {
        // Fallback paths to locate build versions of `byld-lsp`
        const possiblePaths = [
            path.join(context.extensionPath, '..', '..', 'target', 'release', 'byld-lsp'),
            path.join(context.extensionPath, '..', '..', 'target', 'debug', 'byld-lsp'),
            path.join(vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '', 'target', 'release', 'byld-lsp'),
            path.join(vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '', 'target', 'debug', 'byld-lsp'),
        ];

        for (const p of possiblePaths) {
            if (fs.existsSync(p)) {
                serverPath = p;
                break;
            }
        }
    }

    const command = serverPath || 'byld-lsp';

    const run: Executable = {
        command,
        options: {
            env: {
                ...process.env,
                RUST_BACKTRACE: '1',
            }
        }
    };

    const serverOptions: ServerOptions = {
        run,
        debug: run,
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'byld' }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.byd')
        }
    };

    client = new LanguageClient(
        'byldLsp',
        'Byld Language Server',
        serverOptions,
        clientOptions
    );

    client.start();

    // Register Document Color Provider for inline colors and color picker support
    const colorProvider = vscode.languages.registerColorProvider('byld', {
        provideDocumentColors(document) {
            const colors: vscode.ColorInformation[] = [];
            const text = document.getText();
            const regex = /\b0x([0-9a-fA-F]{6}|[0-9a-fA-F]{8})\b/g;
            let match;
            while ((match = regex.exec(text)) !== null) {
                const hex = match[1];
                const startPos = document.positionAt(match.index);
                const endPos = document.positionAt(match.index + match[0].length);
                const range = new vscode.Range(startPos, endPos);
                
                let red = 0, green = 0, blue = 0, alpha = 1.0;
                if (hex.length === 6) {
                    red = parseInt(hex.slice(0, 2), 16) / 255;
                    green = parseInt(hex.slice(2, 4), 16) / 255;
                    blue = parseInt(hex.slice(4, 6), 16) / 255;
                } else if (hex.length === 8) {
                    alpha = parseInt(hex.slice(0, 2), 16) / 255;
                    red = parseInt(hex.slice(2, 4), 16) / 255;
                    green = parseInt(hex.slice(4, 6), 16) / 255;
                    blue = parseInt(hex.slice(6, 8), 16) / 255;
                }
                
                colors.push(new vscode.ColorInformation(range, new vscode.Color(red, green, blue, alpha)));
            }
            return colors;
        },
        provideColorPresentations(color, context) {
            const r = Math.round(color.red * 255).toString(16).padStart(2, '0').toUpperCase();
            const g = Math.round(color.green * 255).toString(16).padStart(2, '0').toUpperCase();
            const b = Math.round(color.blue * 255).toString(16).padStart(2, '0').toUpperCase();
            const a = Math.round(color.alpha * 255).toString(16).padStart(2, '0').toUpperCase();
            
            let label = `0x${r}${g}${b}`;
            // Preserve 8-digit format if it was written or if alpha has transparency
            if (color.alpha < 1.0 || (context.range.end.character - context.range.start.character) === 10) {
                label = `0x${a}${r}${g}${b}`;
            }
            return [new vscode.ColorPresentation(label)];
        }
    });

    context.subscriptions.push(colorProvider);
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
