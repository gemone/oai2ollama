package utils

import (
	"context"
	"crypto/tls"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/gofiber/fiber/v2/log"
)

var (
	protocolCache = sync.Map{}
)

// NewHTTPClient 创建自动优化的HTTP客户端
// 会自动检测协议支持并返回最优配置的客户端
func NewHTTPClient(baseURL string, timeout time.Duration) *http.Client {
	// 先检查缓存
	if cached, ok := protocolCache.Load(baseURL); ok {
		supportsHTTP2 := cached.(bool)
		return createClient(supportsHTTP2, timeout)
	}

	// 缓存未命中，执行检测
	supportsHTTP2 := detectHTTP2Support(baseURL, timeout/4)
	protocolCache.Store(baseURL, supportsHTTP2)

	return createClient(supportsHTTP2, timeout)
}

func createClient(supportsHTTP2 bool, timeout time.Duration) *http.Client {
	// 根据协议支持创建最优配置
	transport := &http.Transport{
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: false,
			MinVersion:         tls.VersionTLS12,
		},
		DialContext: (&net.Dialer{
			Timeout:   timeout,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		ForceAttemptHTTP2:     supportsHTTP2,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   20,
		MaxConnsPerHost:       50,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	}

	return &http.Client{
		Transport: transport,
		Timeout:   timeout,
	}
}

// detectHTTP2Support 快速检测HTTP/2支持
func detectHTTP2Support(baseURL string, timeout time.Duration) bool {
	if timeout <= 0 {
		timeout = 3 * time.Second
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "HEAD", baseURL, nil)
	if err != nil {
		return false
	}

	// 使用支持HTTP/2的临时客户端进行检测
	client := &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{
				InsecureSkipVerify: false,
				MinVersion:         tls.VersionTLS12,
			},
			ForceAttemptHTTP2: true,
		},
		Timeout: timeout,
	}

	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			log.Errorf("Failed to close response body: %v", err)
		}
	}()

	// 检查响应是否使用HTTP/2
	return resp.Proto == "HTTP/2.0"
}
