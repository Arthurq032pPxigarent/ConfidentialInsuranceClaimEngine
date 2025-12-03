// contract.ts
import { ethers } from "ethers";
import abiJson from "./abi/UniversalAdapter.json";
import configJson from "./config.json";

export const ABI = (abiJson as any).abi || abiJson;
export const config = configJson;

const retry = async <T>(fn: () => Promise<T>, retries = 3, delay = 1000): Promise<T> => {
  try {
    return await fn();
  } catch (e) {
    if (retries > 0) {
      await new Promise(res => setTimeout(res, delay));
      return retry(fn, retries - 1, delay * 2);
    }
    throw e;
  }
};

const getTestnetProvider = async () => {
  const rpcUrls = [
    "https://sepolia.infura.io/v3/96406da962744120afbe0cf64c8bd7b3",
    "https://rpc.sepolia.org",
    "https://rpc2.sepolia.org",
    "https://eth-sepolia.public.blastapi.io"
  ];
  
  for (const url of rpcUrls) {
    try {
      const provider = new ethers.JsonRpcProvider(url, {
        name: "sepolia",
        chainId: 11155111
      });
      
      const blockNumber = await Promise.race([
        provider.getBlockNumber(),
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error("RPC timeout")), 10000)
        )
      ]);
      
      console.log(`Connected to RPC ${url}, block: ${blockNumber}`);
      return provider;
    } catch (error) {
      console.warn(`RPC ${url} failed: ${error.message}`);
    }
  }
  
  throw new Error("All RPC providers failed");
};

export async function getContractReadOnly() {
  try {
    const provider = await getTestnetProvider();
    const contract = new ethers.Contract(config.contractAddress, ABI, provider);
    
    const code = await retry(() => provider.getCode(config.contractAddress));
    if (code === "0x") {
      return null;
    }
    
    return contract;
  } catch (error) {
    console.error("Failed to create read-only contract:", error);
    return null;
  }
}

export async function getContractWithSigner() {
  if (!(window as any).ethereum) {
    throw new Error("No injected wallet");
  }
  try {
    const provider = new ethers.BrowserProvider((window as any).ethereum);
    const signer = await provider.getSigner();
    const contract = new ethers.Contract(config.contractAddress, ABI, signer);
    return contract;
  } catch (error) {
    console.error("Failed to create contract with signer:", error);
    throw error;
  }
}

export function normAddr(a: string) { 
  return a ? a.toLowerCase() : a; 
}