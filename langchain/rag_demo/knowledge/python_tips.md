# Python 实用技巧

## 列表推导式
列表推导式是 Python 中创建列表的简洁方式。语法为 [expression for item in iterable if condition]。
例如：squares = [x**2 for x in range(10)] 可以快速生成平方数列表。

## 字典合并
Python 3.9+ 支持用 | 运算符合并字典：merged = dict1 | dict2。
旧版本可以用 {**dict1, **dict2} 的方式合并。

## match-case 语法
Python 3.10 引入了 match-case 结构化模式匹配，类似其他语言的 switch-case，但功能更强大。
支持解构匹配、守卫条件、通配符等高级用法。

## 异步编程
Python 的 asyncio 模块提供了异步编程支持。使用 async/await 语法可以编写高效的并发代码。
适用于 IO 密集型任务，如网络请求、文件读写等。不适合 CPU 密集型任务。
