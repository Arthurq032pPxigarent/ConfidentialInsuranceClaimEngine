// src/types.d.ts
interface Window {
  ethereum?: {
    isMetaMask?: boolean;
    isOkxWallet?: boolean;
    isTrust?: boolean;
    isCoinbaseWallet?: boolean;
    request: (request: { method: string; params?: any[] }) => Promise<any>;
    on: (event: string, handler: (...args: any[]) => void) => void;
    removeListener: (event: string, handler: (...args: any[]) => void) => void;
    selectedAddress?: string;
    chainId?: string;
  };
}