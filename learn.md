## MCP协议

### stdin(standrad input) / stdout(standrad output) —— JSON-RPC

这种是用于本地mcp的，stdin / stdout 是进程间传递用的（实例： echo "abc" | grep "b"），JSON-RPC是协议，是用RPC传输JSON数据

#### tips 为什么要用JSON，而不是二进制RPC，二进制不是更快吗

二进制比JSON的速度可能快几微秒，对于LLM龟速回答来说，换成二进制RPC在用户体验上没有差别
但是对开发和运维而言，JSON人类可直接读取，开发、调试、运维都比二进制好多了，所以使用JSON-RPC
如果想改善用户体验，应该是加速LLM回答，而不是抠这几微妙

### HTTP（客户端） —— SSE（服务端）

客户端使用HTTP发送MCP请求，服务端用SSE回复

#### 服务端为什么用SSE而不是HTTP呢？

因为流式响应或者需要多次往客户端发送信息，所以使用SSE

## Skill

看下来就是写多个prompt模板，然后给LLM name、description和path（注入system prompt），LLM根据用户提问选择合适的skill模板，这个做法是为了更贴合用户的问题，减少上下文并提高上下文关联度

## Memory    → 记住做过什么

memory是面向个人的，会随着对话不断深入

## RAG       → 补充知识

文档是死的，只能人为补充，是面向公众的东西

## Guardrails → 限制不能做什么



## Evaluation → 做得好不好

用框架

## 缓存

## 监控

## HITL

## OpenClaw学习

### Skill

#### frontmatter

用`---`做分隔符，是skill结构化的元数据(metadata)，下面的内容才是skill正文(data)

#### Full模式

skill数量/字符数再预算内时，加载更多的元数据到system prompt，包含`name、description、location`

#### Compact模式

skill太多超出预算时，只保留`name、location`，丢弃`description`

默认full模式最多150个skill、30000字符，超出降级到compact，再超就二分查找裁剪skill数量