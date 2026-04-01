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