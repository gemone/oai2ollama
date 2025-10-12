package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"runtime"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"
	"github.com/gemone/oai2ollama/internal/config"
	"github.com/gemone/oai2ollama/internal/handlers"
)

type CLIOptions struct {
	ConfigPath string
	Port      int
	Host      string
	Debug     bool
	Version   bool
	Help      bool
}

func main() {
	var opts CLIOptions

	// Define command line flags
	flag.StringVar(&opts.ConfigPath, "config", "", "Path to configuration file")
	flag.StringVar(&opts.Host, "host", "", "Server host")
	flag.IntVar(&opts.Port, "port", 0, "Server port")
	flag.BoolVar(&opts.Debug, "debug", false, "Enable debug mode")
	flag.BoolVar(&opts.Version, "version", false, "Show version information")
	flag.BoolVar(&opts.Help, "help", false, "Show help information")

	// Custom help message
	flag.Usage = func() {
		fmt.Printf("oai2ollama-go - OpenAI to Ollama API compatibility layer\n\n")
		fmt.Printf("Usage: %s [options]\n\n", os.Args[0])
		fmt.Printf("Options:\n")
		flag.PrintDefaults()
		fmt.Printf("\nCommands:\n")
		fmt.Printf("  start     Start the server (default)\n")
		fmt.Printf("  config    Open configuration file in editor\n")
		fmt.Printf("  version   Show version information\n")
		fmt.Printf("  help      Show this help message\n")
	}

	flag.Parse()

	// Handle help and version flags
	if opts.Help {
		flag.Usage()
		return
	}

	if opts.Version {
		fmt.Printf("oai2ollama-go version 1.0.0\n")
		fmt.Printf("GitHub: https://github.com/gemone/oai2ollama\n")
		return
	}

	// Handle special commands
	if len(flag.Args()) > 0 {
		command := flag.Args()[0]
		switch command {
		case "config":
			openConfigEditor()
			return
		case "version":
			fmt.Printf("oai2ollama-go version 1.0.0\n")
			return
		case "help":
			flag.Usage()
			return
		default:
			fmt.Printf("Unknown command: %s\n", command)
			fmt.Printf("Run '%s help' for usage information\n", os.Args[0])
			os.Exit(1)
		}
	}

	// Load configuration
	cfg, err := config.Load(opts.ConfigPath)
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// Override config with CLI arguments
	if opts.Host != "" {
		cfg.Server.Host = opts.Host
	}
	if opts.Port != 0 {
		cfg.Server.Port = opts.Port
	}
	if opts.Debug {
		cfg.Server.Debug = true
	}

	// Create Fiber app with debug mode
	app := fiber.New(fiber.Config{
		ErrorHandler: func(c *fiber.Ctx, err error) error {
			log.Printf("[ERROR] Fiber Error Handler: %v (Type: %T)", err, err)

			// In debug mode, print stack trace
			if cfg.Server.Debug {
				log.Printf("[ERROR] Stack Trace: %s", getStackTrace())
			}

			code := fiber.StatusInternalServerError
			if e, ok := err.(*fiber.Error); ok {
				code = e.Code
			}

			// In debug mode, return more detailed error info
			if cfg.Server.Debug {
				return c.Status(code).JSON(map[string]interface{}{
					"error": map[string]interface{}{
						"message":     err.Error(),
						"type":        "internal_error",
						"code":        "unknown_error",
						"debug":       true,
						"stack_trace": getStackTrace(),
					},
				})
			}

			return c.Status(code).JSON(map[string]interface{}{
				"error": map[string]interface{}{
					"message": err.Error(),
					"type":    "internal_error",
					"code":    "unknown_error",
				},
			})
		},
		EnablePrintRoutes: true,
	})

	// Add middleware with custom logger configuration
	app.Use(recover.New(recover.Config{
		EnableStackTrace: true,
		StackTraceHandler: func(c *fiber.Ctx, e interface{}) {
			log.Printf("[PANIC] Stack trace: %v", e)
		},
	}))

	app.Use(logger.New(logger.Config{
		Format: "${time} | ${status} | ${latency} | ${ip} | ${method} | ${path} | ${error}\n",
		Output: os.Stdout,
	}))
	app.Use(cors.New(cors.Config{
		AllowOrigins: "*",
		AllowMethods: "GET,POST,PUT,DELETE,OPTIONS",
		AllowHeaders: "Origin,Content-Type,Accept,Authorization",
	}))

	// Initialize handlers
	ollamaHandler := handlers.NewOllamaHandler(cfg)

	// Setup routes
	setupRoutes(app, ollamaHandler)

	// Health check endpoint
	app.Get("/", func(c *fiber.Ctx) error {
		return c.JSON(map[string]interface{}{
			"name":        "oai2ollama-go",
			"version":     "1.0.0",
			"description": "OpenAI to Ollama API compatibility layer",
			"config":      opts.ConfigPath,
			"endpoints": map[string]interface{}{
				"ollama_api": map[string]string{
					"generate":    "POST /api/generate",
					"chat":        "POST /api/chat",
					"embeddings":  "POST /api/embeddings",
					"tags":        "GET /api/tags",
					"show":        "POST /api/show",
					"ps":          "GET /api/ps",
					"pull":        "POST /api/pull",
					"delete":      "DELETE /api/delete",
					"copy":        "POST /api/copy",
					"create":      "POST /api/create",
					"version":     "GET /api/version",
				},
			},
		})
	})

	app.Head("/", func(c *fiber.Ctx) error {
		return c.SendStatus(200)
	})

	// Start server
	host := cfg.Server.Host
	port := cfg.Server.Port
	addr := fmt.Sprintf("%s:%d", host, port)

	log.Printf("Starting oai2ollama-go server on %s", addr)
	if opts.ConfigPath != "" {
		log.Printf("Using config file: %s", opts.ConfigPath)
	}
	log.Printf("Health check: http://%s/", addr)
	log.Printf("API docs: http://%s/api", addr)

	if err := app.Listen(addr); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

func setupRoutes(app *fiber.App, ollamaHandler *handlers.OllamaHandler) {
	// Ollama API compatible endpoints
	api := app.Group("/api")

	// Core Ollama API endpoints
	api.Get("/tags", ollamaHandler.ListModels)           // List models
	api.Post("/chat", ollamaHandler.Chat)                // Chat completion
	api.Post("/generate", ollamaHandler.Generate)        // Generate completion
	api.Post("/embeddings", ollamaHandler.Embeddings)    // Generate embeddings
	api.Get("/version", ollamaHandler.Version)           // Get version
	api.Get("/ps", ollamaHandler.ShowRunningModels)      // Show running models
	api.Post("/pull", ollamaHandler.PullModel)           // Pull model (mock)
	api.Post("/show", ollamaHandler.ShowModel)           // Show model information
	api.Delete("/delete", ollamaHandler.DeleteModel)     // Delete model (mock)
}

func openConfigEditor() {
	configPath := "configs/config.yaml"
	if len(os.Args) > 2 {
		configPath = os.Args[2]
	}

	// Try different editors
	editors := []string{"code", "nano", "vim", "vi"}
	var cmd string
	var args []string

	for _, editor := range editors {
		_, err := os.Stat("/usr/bin/" + editor)
		if err == nil {
			cmd = editor
			args = []string{configPath}
			break
		}
	}

	if cmd == "" {
		fmt.Printf("No supported editor found. Please edit %s manually.\n", configPath)
		return
	}

	fmt.Printf("Opening %s in %s...\n", configPath, cmd)
	execCmd := exec.Command(cmd, args...)
	execCmd.Stdout = os.Stdout
	execCmd.Stderr = os.Stderr
	execCmd.Run()
}

// getStackTrace returns the current stack trace as a string
func getStackTrace() string {
	buf := make([]byte, 1024)
	for {
		n := runtime.Stack(buf, false)
		if n < len(buf) {
			return string(buf[:n])
		}
		buf = make([]byte, 2*len(buf))
	}
}

// logDebug logs debug information if debug mode is enabled
func logDebug(debug bool, format string, args ...interface{}) {
	if debug {
		log.Printf("[DEBUG] %s", fmt.Sprintf(format, args...))
	}
}