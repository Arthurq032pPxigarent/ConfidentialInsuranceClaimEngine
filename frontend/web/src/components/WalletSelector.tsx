import React, { useState, useEffect } from 'react';

interface WalletInfo {
  name: string;
  provider: any;
  icon: string;
  isInstalled: boolean;
}

interface WalletSelectorProps {
  isOpen: boolean;
  onWalletSelect: (wallet: WalletInfo) => void;
  onClose: () => void;
}

const WalletSelector: React.FC<WalletSelectorProps> = ({ isOpen, onWalletSelect, onClose }) => {
  const [availableWallets, setAvailableWallets] = useState<WalletInfo[]>([]);
  const [showOtherWallets, setShowOtherWallets] = useState(false);

  useEffect(() => {
    detectWallets();
  }, []);

  const detectWallets = () => {
    const wallets: WalletInfo[] = [];

    const walletIcons: Record<string, string> = {
      'MetaMask': 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/MetaMask_Fox.svg/1200px-MetaMask_Fox.svg.png',
      'OKX Wallet': 'https://www.okx.com/favicon.ico',
      'Binance Wallet': 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/Binance_Logo.png/600px-Binance_Logo.png',
      'Other Wallets': 'https://cdn-icons-png.flaticon.com/512/126/126472.png'
    };

    // MetaMask
    if (typeof (window as any).ethereum !== 'undefined' && (window as any).ethereum.isMetaMask) {
      wallets.push({
        name: 'MetaMask',
        provider: (window as any).ethereum,
        icon: walletIcons['MetaMask'],
        isInstalled: true
      });
    } else {
      wallets.push({
        name: 'MetaMask',
        provider: null,
        icon: walletIcons['MetaMask'],
        isInstalled: false
      });
    }

    // OKX Wallet
    if (typeof (window as any).okxwallet !== 'undefined') {
      wallets.push({
        name: 'OKX Wallet',
        provider: (window as any).okxwallet,
        icon: walletIcons['OKX Wallet'],
        isInstalled: true
      });
    } else {
      wallets.push({
        name: 'OKX Wallet',
        provider: null,
        icon: walletIcons['OKX Wallet'],
        isInstalled: false
      });
    }

    // Binance Wallet
    if (typeof (window as any).BinanceChain !== 'undefined') {
      wallets.push({
        name: 'Binance Wallet',
        provider: (window as any).BinanceChain,
        icon: walletIcons['Binance Wallet'],
        isInstalled: true
      });
    } else {
      wallets.push({
        name: 'Binance Wallet',
        provider: null,
        icon: walletIcons['Binance Wallet'],
        isInstalled: false
      });
    }

    wallets.push({
      name: 'Other Wallets',
      provider: null,
      icon: walletIcons['Other Wallets'],
      isInstalled: true
    });

    setAvailableWallets(wallets);
  };

  const handleWalletSelect = async (wallet: WalletInfo) => {
    if (wallet.name === 'Other Wallets') {
      setShowOtherWallets(true);
      return;
    }

    if (!wallet.isInstalled) {
      // Open wallet download page
      const walletUrls: { [key: string]: string } = {
        'MetaMask': 'https://metamask.io/',
        'OKX Wallet': 'https://www.okx.com/web3',
        'Binance Wallet': 'https://www.bnbchain.org/en/binance-wallet',
      };
      
      const url = walletUrls[wallet.name];
      if (url) {
        window.open(url, '_blank');
      }
      return;
    }

    try {
      // Auto-switch to Sepolia testnet
      await switchToSepolia(wallet.provider);
      onWalletSelect(wallet);
    } catch (error) {
      console.error('Error switching network:', error);
      // Continue anyway, let the main app handle the connection
      onWalletSelect(wallet);
    }
  };

  const switchToSepolia = async (provider: any) => {
    try {
      await provider.request({
        method: 'wallet_switchEthereumChain',
        params: [{ chainId: '0xaa36a7' }], // Sepolia chainId
      });
    } catch (switchError: any) {
      if (switchError.code === 4902) {
        try {
          await provider.request({
            method: 'wallet_addEthereumChain',
            params: [{
              chainId: '0xaa36a7',
              chainName: 'Sepolia',
              nativeCurrency: {
                name: 'Sepolia Ether',
                symbol: 'SEP',
                decimals: 18
              },
              rpcUrls: [
                'https://rpc.sepolia.org',
                'https://eth-sepolia.public.blastapi.io',
              ],
              blockExplorerUrls: ['https://sepolia.etherscan.io/']
            }]
          });
        } catch (addError) {
          console.error('Error adding Sepolia network:', addError);
          // Don't throw, let the main app handle it
        }
      }
    }
  };

  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0, 0, 0, 0.7)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      backdropFilter: 'blur(5px)',
      fontFamily: "'Inter', sans-serif"
    }}>
      <div style={{
        background: 'linear-gradient(135deg, #1a1c2c 0%, #2d325a 100%)',
        borderRadius: '20px',
        padding: '24px',
        width: '90%',
        maxWidth: '350px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        boxShadow: '0 20px 40px rgba(0, 0, 0, 0.2)',
        color: 'white',
        overflow: 'hidden',
        position: 'relative'
      }}>
        {showOtherWallets && (
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.8)',
            zIndex: 10,
            display: 'flex',
            flexDirection: 'column',
            padding: '20px'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h3 style={{ margin: 0, fontSize: '18px', fontWeight: '600' }}>Other Wallets</h3>
              <button
                onClick={() => setShowOtherWallets(false)}
                style={{
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: 'none',
                  color: 'white',
                  width: '30px',
                  height: '30px',
                  borderRadius: '50%',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '16px'
                }}
              >
                ✕
              </button>
            </div>
            
            <div style={{ flex: 1, overflowY: 'auto' }}>
              <div style={{ 
                background: 'rgba(255, 255, 255, 0.05)', 
                padding: '15px', 
                borderRadius: '12px',
                marginBottom: '15px'
              }}>
                <div style={{ fontWeight: '500', marginBottom: '10px' }}>Popular Wallets</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
                  {['Coinbase Wallet', 'Trust Wallet', 'WalletConnect', 'Ledger', 'Trezor'].map(name => (
                    <div key={name} style={{
                      background: 'rgba(255, 255, 255, 0.1)',
                      padding: '8px 12px',
                      borderRadius: '20px',
                      fontSize: '14px'
                    }}>
                      {name}
                    </div>
                  ))}
                </div>
              </div>
              
              <div style={{ 
                background: 'rgba(255, 255, 255, 0.05)', 
                padding: '15px', 
                borderRadius: '12px'
              }}>
                <div style={{ fontWeight: '500', marginBottom: '10px' }}>Mobile Wallets</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
                  {['TokenPocket', 'MathWallet', 'SafePal', 'BitKeep', 'ImToken'].map(name => (
                    <div key={name} style={{
                      background: 'rgba(255, 255, 255, 0.1)',
                      padding: '8px 12px',
                      borderRadius: '20px',
                      fontSize: '14px'
                    }}>
                      {name}
                    </div>
                  ))}
                </div>
              </div>
            </div>
            
            <div style={{ marginTop: '15px', textAlign: 'center' }}>
              <button
                onClick={() => setShowOtherWallets(false)}
                style={{
                  background: 'linear-gradient(90deg, #8a2be2, #00bfff)',
                  color: 'white',
                  border: 'none',
                  padding: '10px 20px',
                  borderRadius: '30px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  width: '100%'
                }}
              >
                Back to Main
              </button>
            </div>
          </div>
        )}
        
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', position: 'relative', zIndex: 1 }}>
          <h2 style={{ 
            margin: 0, 
            fontSize: '22px', 
            fontWeight: '700',
            background: 'linear-gradient(90deg, #8a2be2, #00bfff)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent'
          }}>
            Connect Wallet
          </h2>
          <button
            onClick={onClose}
            style={{
              background: 'rgba(255, 255, 255, 0.1)',
              border: 'none',
              color: 'white',
              width: '36px',
              height: '36px',
              borderRadius: '50%',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s ease',
              fontSize: '18px'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.2)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
            }}
          >
            ✕
          </button>
        </div>
        
        <div style={{ 
          display: 'flex', 
          flexDirection: 'column',
          gap: '12px',
          marginBottom: '24px',
          position: 'relative',
          zIndex: 1
        }}>
          {availableWallets.map((wallet, index) => (
            <div
              key={index}
              onClick={() => handleWalletSelect(wallet)}
              style={{
                borderRadius: '12px',
                border: wallet.isInstalled 
                  ? '1px solid rgba(255, 255, 255, 0.15)' 
                  : '1px solid rgba(255, 255, 255, 0.08)',
                background: wallet.isInstalled 
                  ? 'rgba(255, 255, 255, 0.05)' 
                  : 'rgba(255, 255, 255, 0.03)',
                padding: '16px',
                display: 'flex',
                alignItems: 'center',
                gap: '16px',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                position: 'relative',
                overflow: 'hidden'
              }}
              onMouseOver={(e) => {
                if (wallet.isInstalled) {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)';
                  e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.2)';
                  e.currentTarget.style.transform = 'translateY(-2px)';
                }
              }}
              onMouseOut={(e) => {
                if (wallet.isInstalled) {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                  e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.15)';
                  e.currentTarget.style.transform = 'translateY(0)';
                }
              }}
            >
              {/* Wallet icon */}
              <div style={{
                width: '48px',
                height: '48px',
                borderRadius: '12px',
                background: 'rgba(255, 255, 255, 0.1)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '8px',
                flexShrink: 0
              }}>
                <img 
                  src={wallet.icon} 
                  alt={wallet.name} 
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain'
                  }}
                  onError={(e) => {
                    // Fallback if image fails to load
                    const target = e.target as HTMLImageElement;
                    target.src = 'https://via.placeholder.com/48';
                  }}
                />
              </div>
              
              {/* Wallet name and status */}
              <div style={{ flex: 1 }}>
                <div style={{ 
                  fontWeight: '500', 
                  fontSize: '16px',
                  marginBottom: '4px'
                }}>
                  {wallet.name}
                </div>
                <div style={{
                  fontSize: '14px',
                  opacity: 0.7
                }}>
                  {wallet.isInstalled ? 'Ready to connect' : 'Click to install'}
                </div>
              </div>
              
              {/* Status indicator */}
              <div style={{
                width: '12px',
                height: '12px',
                borderRadius: '50%',
                background: wallet.isInstalled ? '#4CAF50' : '#FF9800',
                flexShrink: 0
              }}></div>
            </div>
          ))}
        </div>
        
        <div style={{
          fontSize: '13px',
          opacity: 0.7,
          textAlign: 'center',
          fontWeight: '500',
          position: 'relative',
          zIndex: 1,
          padding: '12px',
          background: 'rgba(255, 255, 255, 0.05)',
          borderRadius: '12px',
          border: '1px solid rgba(255, 255, 255, 0.08)'
        }}>
          <div style={{ marginBottom: '6px' }}>
            Wallet will automatically switch to Sepolia testnet
          </div>
        </div>
      </div>
    </div>
  );
};

export default WalletSelector;