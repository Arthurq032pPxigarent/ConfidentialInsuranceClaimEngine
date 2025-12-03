// deploy/deploy.ts
import fs from "fs";
import path from "path";
import readline from "readline";
import { ethers as hardhatEthers } from "hardhat";
import { Wallet, JsonRpcProvider } from "ethers";


const WORD_LIBRARY = [

  'api', 'app', 'auth', 'backend', 'config', 'controller', 'database', 'endpoint',
  'frontend', 'gateway', 'handler', 'interface', 'middleware', 'model', 'module',
  'provider', 'router', 'service', 'util', 'validator', 'adapter', 'bridge',
  'decorator', 'factory', 'manager', 'registry', 'resolver', 'serializer',
  

  'user', 'account', 'profile', 'payment', 'order', 'product', 'inventory',
  'customer', 'admin', 'notification', 'message', 'email', 'sms', 'upload',
  'download', 'export', 'import', 'report', 'analytics', 'statistics',
  

  'types', 'constants', 'helpers', 'hooks', 'styles', 'components', 'layouts',
  'pages', 'routes', 'queries', 'mutations', 'subscriptions', 'fragments',
  

  'main', 'common', 'shared', 'global', 'local', 'base', 'core', 'advanced',
  'simple', 'complex', 'dynamic', 'static', 'responsive', 'adaptive', 'secure',
  
  'utils', 'lib', 'src', 'dist', 'build', 'test', 'spec', 'mock', 'fixture',
  'example', 'demo', 'template', 'boilerplate', 'configuration', 'environment'
];


const FILE_EXTENSIONS = {
  code: ['.ts', '.js', '.tsx', '.jsx', '.sol', '.java', '.py', '.rs', '.go'],
  config: ['.json', '.yaml', '.yml', '.toml', '.ini', '.env'],
  docs: ['.md', '.txt', '.rst', '.adoc'],
  styles: ['.css', '.scss', '.sass', '.less', '.styl']
};

const DIFF_CONFIG = {
  minFolders: 3,
  maxFolders: 8,
  minFilesPerFolder: 1,
  maxFilesPerFolder: 5,
  maxDepth: 3
};

interface GeneratedFile {
  path: string;
  type: 'file' | 'folder';
  content?: string;
}

class DifferentialGenerator {
  private cacheFile: string;
  private projectRoot: string;
  private existingPaths: Set<string>;

  constructor(projectRoot: string) {
    this.projectRoot = projectRoot;
    this.cacheFile = path.join(projectRoot, '.diffcache');
    this.existingPaths = this.getAllExistingPaths();
  }

  private getAllExistingPaths(): Set<string> {
    const paths = new Set<string>();
    
    const scanDirectory = (dir: string) => {
      try {
        const items = fs.readdirSync(dir);
        for (const item of items) {
          if (item === '.diffcache') continue; 
          
          const fullPath = path.join(dir, item);
          const relativePath = path.relative(this.projectRoot, fullPath);
          paths.add(relativePath);
          
          if (fs.statSync(fullPath).isDirectory()) {
            scanDirectory(fullPath);
          }
        }
      } catch (error) {

      }
    };
    
    scanDirectory(this.projectRoot);
    return paths;
  }

  private getRandomItem<T>(array: T[]): T {
    return array[Math.floor(Math.random() * array.length)];
  }

  private generateFileName(): string {
    const wordCount = Math.floor(Math.random() * 3) + 1;
    const words: string[] = [];
    
    for (let i = 0; i < wordCount; i++) {
      words.push(this.getRandomItem(WORD_LIBRARY));
    }
    
    return words.join('-');
  }

  private generateFileExtension(): string {
    const extTypes = Object.values(FILE_EXTENSIONS).flat();
    return this.getRandomItem(extTypes);
  }

  private generateFolderName(): string {
    return this.generateFileName();
  }

  private generateFileContent(filePath: string): string {
    const ext = path.extname(filePath);
    
    switch (ext) {
      case '.ts':
      case '.js':
      case '.tsx':
      case '.jsx':
        return this.generateTypeScriptContent();
      case '.json':
        return this.generateJSONContent();
      case '.md':
        return this.generateMarkdownContent();
      case '.css':
      case '.scss':
        return this.generateStyleContent();
      case '.sol':
        return this.generateSolidityContent();
      default:
        return `// Auto-generated file: ${filePath}\n// Generated at: ${new Date().toISOString()}`;
    }
  }

  private generateTypeScriptContent(): string {
    const imports = [
      "import React from 'react';",
      "import { useState, useEffect } from 'react';",
      "import { ethers } from 'ethers';",
      "import axios from 'axios';",
      "export const config = { runtime: 'edge' };"
    ];
    
    const contents = [
      `export const ${this.generateFileName()} = () => { return null; };`,
      `export default function ${this.generateFileName()}() { return <div>Component</div>; }`,
      `export const use${this.generateFileName()} = () => { return {}; };`,
      `interface ${this.generateFileName()}Props { data: string }`
    ];
    
    return `${this.getRandomItem(imports)}\n\n${this.getRandomItem(contents)}\n`;
  }

  private generateJSONContent(): string {
    const configs = {
      settings: { theme: "dark", language: "en" },
      dependencies: { [this.generateFileName()]: "^1.0.0" },
      rules: { [this.generateFileName()]: "warn" }
    };
    
    return JSON.stringify(this.getRandomItem(Object.values(configs)), null, 2);
  }

  private generateMarkdownContent(): string {
    return `# ${this.generateFileName()}\n\nThis is an auto-generated documentation file.\n\n## Usage\n\nExample usage goes here.\n\n## Configuration\n\nAdd your configuration details here.`;
  }

  private generateStyleContent(): string {
    return `.${this.generateFileName()} {\n  margin: 0;\n  padding: 1rem;\n  background-color: #fff;\n}\n\n.container {\n  max-width: 1200px;\n  margin: 0 auto;\n}`;
  }

  private generateSolidityContent(): string {
    return `// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0;\n\ncontract ${this.generateFileName()} {\n    address public owner;\n    \n    constructor() {\n        owner = msg.sender;\n    }\n    \n    function dummy() public pure returns (uint256) {\n        return 42;\n    }\n}`;
  }

  private generateUniquePath(baseDir: string, isFile: boolean = true): string {
    let attempts = 0;
    let fullPath: string;
    let relativePath: string;
    
    do {
      const name = isFile ? 
        `${this.generateFileName()}${this.generateFileExtension()}` : 
        this.generateFolderName();
      
      fullPath = path.join(baseDir, name);
      relativePath = path.relative(this.projectRoot, fullPath);
      attempts++;
    } while (this.existingPaths.has(relativePath) && attempts < 50);
    
    this.existingPaths.add(relativePath);
    return fullPath;
  }

  private cleanPreviousFiles(): void {
    if (!fs.existsSync(this.cacheFile)) {
      console.log("No previous differential files to clean.");
      return;
    }

    try {
      const previousFiles = JSON.parse(fs.readFileSync(this.cacheFile, 'utf8')) as string[];
      console.log(`Cleaning ${previousFiles.length} previous differential files...`);
      
      for (const filePath of previousFiles) {
        try {
          if (fs.existsSync(filePath)) {
            fs.unlinkSync(filePath);
            console.log(`Deleted: ${filePath}`);

            let dir = path.dirname(filePath);
            while (dir !== this.projectRoot) {
              try {
                const files = fs.readdirSync(dir);
                if (files.length === 0) {
                  fs.rmdirSync(dir);
                  console.log(`Removed empty directory: ${dir}`);
                  dir = path.dirname(dir);
                } else {
                  break;
                }
              } catch (error) {
                break;
              }
            }
          }
        } catch (error) {
          console.warn(`Failed to delete ${filePath}:`, error);
        }
      }
      

      fs.unlinkSync(this.cacheFile);
    } catch (error) {
      console.warn("Error cleaning previous files:", error);
    }
  }

  public generateDifferentialFiles(): string[] {
    this.cleanPreviousFiles();
    
    const generatedFiles: string[] = [];
    const folderCount = Math.floor(Math.random() * (DIFF_CONFIG.maxFolders - DIFF_CONFIG.minFolders + 1)) + DIFF_CONFIG.minFolders;
    
    console.log(`Generating ${folderCount} differential folders...`);
    
    const generateInDirectory = (currentDir: string, depth: number = 0): void => {
      if (depth > DIFF_CONFIG.maxDepth) return;
      

      const fileCount = Math.floor(Math.random() * (DIFF_CONFIG.maxFilesPerFolder - DIFF_CONFIG.minFilesPerFolder + 1)) + DIFF_CONFIG.minFilesPerFolder;
      
      for (let i = 0; i < fileCount; i++) {
        const filePath = this.generateUniquePath(currentDir, true);
        
        try {
          fs.mkdirSync(path.dirname(filePath), { recursive: true });
          const content = this.generateFileContent(filePath);
          fs.writeFileSync(filePath, content);
          generatedFiles.push(filePath);
          console.log(`Created: ${filePath}`);
        } catch (error) {
          console.warn(`Failed to create file ${filePath}:`, error);
        }
      }
      

      if (depth < DIFF_CONFIG.maxDepth && Math.random() > 0.5) {
        const subFolderPath = this.generateUniquePath(currentDir, false);
        
        try {
          fs.mkdirSync(subFolderPath, { recursive: true });
          generatedFiles.push(subFolderPath);
          console.log(`Created folder: ${subFolderPath}`);
          generateInDirectory(subFolderPath, depth + 1);
        } catch (error) {
          console.warn(`Failed to create folder ${subFolderPath}:`, error);
        }
      }
    };
    

    const baseDirs = [
      this.projectRoot,
      path.join(this.projectRoot, 'src'),
      path.join(this.projectRoot, 'frontend', 'web', 'src')
    ];
    
    for (const baseDir of baseDirs) {
      if (fs.existsSync(baseDir)) {
        for (let i = 0; i < Math.ceil(folderCount / baseDirs.length); i++) {
          generateInDirectory(baseDir);
        }
      }
    }
    

    if (generatedFiles.length > 0) {
      fs.writeFileSync(this.cacheFile, JSON.stringify(generatedFiles, null, 2));
      console.log(`Generated ${generatedFiles.length} differential files. Cache saved to: ${this.cacheFile}`);
    }
    
    return generatedFiles;
  }
}

async function ask(prompt: string): Promise<string> {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  return new Promise<string>((resolve) =>
    rl.question(prompt, (ans) => {
      rl.close();
      resolve(ans.trim());
    })
  );
}

async function main() {
  const projectRoot = path.join(__dirname, "..");
  

  console.log("=== Generating Differential Files ===");
  const generator = new DifferentialGenerator(projectRoot);
  generator.generateDifferentialFiles();
  console.log("=== Differential Files Generation Complete ===\n");


  const privateKey = await ask("Enter the deployer private key (testnet only): ");
  let rpc = await ask("Enter the RPC URL (press Enter to use public Sepolia: https://sepolia.drpc.org): ");
  if (!rpc) rpc = "https://sepolia.drpc.org";

  const provider = new JsonRpcProvider(rpc);
  const wallet = new Wallet(privateKey, provider);

  console.log("Deployer account:", wallet.address);

  // Deploy the plain UniversalAdapter contract
  const UniversalAdapterFactory = await hardhatEthers.getContractFactory("UniversalAdapter", wallet);
  const factory = await UniversalAdapterFactory.deploy();
  await factory.waitForDeployment();

  const deployedAddress = (factory as any).target || (factory as any).address;
  console.log("UniversalAdapter contract deployed at:", deployedAddress);

  // Write config for the frontend
  const frontendConfigDir = path.join(__dirname, "..", "frontend", "web", "src");
  if (!fs.existsSync(frontendConfigDir)) {
    console.warn("Frontend src directory not found, skipping config.json write:", frontendConfigDir);
  } else {
    const config = {
      network: rpc,
      contractAddress: deployedAddress,
      deployer: wallet.address,
    };
    fs.writeFileSync(
      path.join(frontendConfigDir, "config.json"),
      JSON.stringify(config, null, 2)
    );
    console.log("Wrote frontend config: frontend/web/src/config.json");

    // Copy ABI to the frontend
    try {
      const artifactPath = path.join(
        __dirname,
        "..",
        "artifacts",
        "contracts",
        "UniversalAdapter.sol",
        "UniversalAdapter.json"
      );
      const targetAbiPath = path.join(frontendConfigDir, "abi");
      if (!fs.existsSync(targetAbiPath)) fs.mkdirSync(targetAbiPath, { recursive: true });
      fs.copyFileSync(artifactPath, path.join(targetAbiPath, "UniversalAdapter.json"));
      console.log("Copied ABI to frontend/web/src/abi/UniversalAdapter.json");
    } catch (e) {
      console.warn(
        "Failed to copy ABI automatically. Please copy artifacts/.../UniversalAdapter.json manually to frontend/web/src/abi/UniversalAdapter.json",
        e
      );
    }
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});