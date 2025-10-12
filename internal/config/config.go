package config

import (
	"fmt"
	"strings"

	"github.com/spf13/viper"
)

type Config struct {
	Server    ServerConfig    `mapstructure:"server"`
	Backends  []BackendConfig `mapstructure:"backends"`
	Models    []ModelConfig   `mapstructure:"models"`
	Logging   LoggingConfig   `mapstructure:"logging"`
	Database  DatabaseConfig  `mapstructure:"database"`
}

type ServerConfig struct {
	Host         string `mapstructure:"host"`
	Port         int    `mapstructure:"port"`
	Debug        bool   `mapstructure:"debug"`
	ReadTimeout  int    `mapstructure:"read_timeout"`
	WriteTimeout int    `mapstructure:"write_timeout"`
}

type BackendConfig struct {
	Name         string             `mapstructure:"name"`
	APIKey       string             `mapstructure:"api_key"`
	BaseURL      string             `mapstructure:"base_url"`
	ModelURL     string             `mapstructure:"model_url"`
	Enabled      bool               `mapstructure:"enabled"`
	Timeout      int                `mapstructure:"timeout"`
	Priority     int                `mapstructure:"priority"`
	ModelPrefix  *ModelPrefixConfig `mapstructure:"model_prefix"`
	DefaultPrompt *DefaultPromptConfig `mapstructure:"default_prompt"`
}

type ModelPrefixConfig struct {
	Enabled   bool   `mapstructure:"enabled"`
	Prefix    string `mapstructure:"prefix"`
	Separator string `mapstructure:"separator"`
	Override  bool   `mapstructure:"override"`
}

type DefaultPromptConfig struct {
	Enabled bool   `mapstructure:"enabled"`
	Prompt  string `mapstructure:"prompt"`
}

type ModelConfig struct {
	Name         string   `mapstructure:"name"`
	Backend      string   `mapstructure:"backend"`
	Capabilities []string `mapstructure:"capabilities"`
	Enabled      bool     `mapstructure:"enabled"`
	OriginalName string   `mapstructure:"original_name"`
	DisplayName  string   `mapstructure:"display_name"`
}

type LoggingConfig struct {
	Level      string `mapstructure:"level"`
	Format     string `mapstructure:"format"`
	File       string `mapstructure:"file"`
	MaxSize    int    `mapstructure:"max_size"`
	MaxBackups int    `mapstructure:"max_backups"`
	MaxAge     int    `mapstructure:"max_age"`
	Compress   bool   `mapstructure:"compress"`
}

type DatabaseConfig struct {
	Path             string `mapstructure:"path"`
	MaxConnections   int    `mapstructure:"max_connections"`
	ConnectionTimeout int    `mapstructure:"connection_timeout"`
}

var GlobalConfig *Config

func Load(configPath string) (*Config, error) {
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")

	if configPath != "" {
		viper.SetConfigFile(configPath)
	} else {
		viper.AddConfigPath("./configs")
		viper.AddConfigPath(".")
		viper.AddConfigPath("$HOME/.oai2ollama")
	}

	// Set defaults
	setDefaults()

	// Enable environment variable support
	viper.AutomaticEnv()
	viper.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))

	// Read config
	if err := viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); ok {
			// Config file not found, use defaults and env vars
			fmt.Printf("Config file not found, using defaults and environment variables\n")
		} else {
			return nil, fmt.Errorf("error reading config file: %w", err)
		}
	}

	var config Config
	if err := viper.Unmarshal(&config); err != nil {
		return nil, fmt.Errorf("error unmarshaling config: %w", err)
	}

	// Validate config
	if err := validate(&config); err != nil {
		return nil, fmt.Errorf("config validation failed: %w", err)
	}

	GlobalConfig = &config
	return &config, nil
}

func setDefaults() {
	// Server defaults
	viper.SetDefault("server.host", "localhost")
	viper.SetDefault("server.port", 11434)
	viper.SetDefault("server.debug", false)
	viper.SetDefault("server.read_timeout", 60)
	viper.SetDefault("server.write_timeout", 60)

	// Logging defaults
	viper.SetDefault("logging.level", "info")
	viper.SetDefault("logging.format", "json")
	viper.SetDefault("logging.file", "logs/app.log")
	viper.SetDefault("logging.max_size", 100)
	viper.SetDefault("logging.max_backups", 7)
	viper.SetDefault("logging.max_age", 30)
	viper.SetDefault("logging.compress", true)

	// Database defaults
	viper.SetDefault("database.path", "metrics.db")
	viper.SetDefault("database.max_connections", 10)
	viper.SetDefault("database.connection_timeout", 30)
}

func validate(config *Config) error {
	// Validate server config
	if config.Server.Port <= 0 || config.Server.Port > 65535 {
		return fmt.Errorf("invalid server port: %d", config.Server.Port)
	}

	// Validate backends
	if len(config.Backends) == 0 {
		return fmt.Errorf("at least one backend must be configured")
	}

	backendNames := make(map[string]bool)
	for i, backend := range config.Backends {
		if backend.Name == "" {
			return fmt.Errorf("backend %d: name cannot be empty", i)
		}
		if backendNames[backend.Name] {
			return fmt.Errorf("duplicate backend name: %s", backend.Name)
		}
		backendNames[backend.Name] = true

		if backend.APIKey == "" {
			return fmt.Errorf("backend %s: api_key cannot be empty", backend.Name)
		}
		if backend.BaseURL == "" {
			return fmt.Errorf("backend %s: base_url cannot be empty", backend.Name)
		}
		if backend.Timeout <= 0 {
			backend.Timeout = 30 // Default timeout
		}
		if backend.Priority <= 0 {
			backend.Priority = 100 // Default priority
		}

		// Set default model URL if not provided
		if backend.ModelURL == "" {
			backend.ModelURL = backend.BaseURL + "/models"
		}

		// Set default prefix config if not provided
		if backend.ModelPrefix != nil && backend.ModelPrefix.Prefix == "" {
			backend.ModelPrefix.Prefix = backend.Name
		}
	}

	// Validate models
	for i, model := range config.Models {
		if model.Name == "" {
			return fmt.Errorf("model %d: name cannot be empty", i)
		}
		if model.Backend == "" {
			return fmt.Errorf("model %s: backend cannot be empty", model.Name)
		}
		if !backendNames[model.Backend] {
			return fmt.Errorf("model %s: backend '%s' not found", model.Name, model.Backend)
		}
	}

	return nil
}

func GetBackend(name string) (*BackendConfig, bool) {
	if GlobalConfig == nil {
		return nil, false
	}

	for _, backend := range GlobalConfig.Backends {
		if backend.Name == name && backend.Enabled {
			return &backend, true
		}
	}
	return nil, false
}

func GetEnabledBackends() []BackendConfig {
	if GlobalConfig == nil {
		return nil
	}

	var enabled []BackendConfig
	for _, backend := range GlobalConfig.Backends {
		if backend.Enabled {
			enabled = append(enabled, backend)
		}
	}
	return enabled
}

func GetDefaultPromptForBackend(backendName string) (string, bool) {
	if GlobalConfig == nil {
		return "", false
	}

	for _, backend := range GlobalConfig.Backends {
		if backend.Name == backendName && backend.Enabled && backend.DefaultPrompt != nil {
			if backend.DefaultPrompt.Enabled && backend.DefaultPrompt.Prompt != "" {
				return backend.DefaultPrompt.Prompt, true
			}
		}
	}
	return "", false
}