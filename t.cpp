#include <bits/stdc++.h>

using namespace std;

struct BigInt
{
	static const int BASE = 1000000000; // 压9位
	vector<int> nodes;

	// 构造函数
	BigInt(long long v = 0)
	{
		if (v == 0)
			nodes.push_back(0);
		while (v > 0)
		{
			nodes.push_back(v % BASE);
			v /= BASE;
		}
	}

	BigInt(string s)
	{
		for (int i = s.size(); i > 0; i -= 9)
		{
			if (i < 9)
				nodes.push_back(stoi(s.substr(0, i)));
			else
				nodes.push_back(stoi(s.substr(i - 9, 9)));
		}
		trim();
	}

	void trim()
	{
		while (nodes.size() > 1 && nodes.back() == 0)
			nodes.pop_back();
	}

	// 比较运算符
	bool operator<(const BigInt &b) const
	{
		if (nodes.size() != b.nodes.size())
			return nodes.size() < b.nodes.size();
		for (int i = nodes.size() - 1; i >= 0; i--)
			if (nodes[i] != b.nodes[i])
				return nodes[i] < b.nodes[i];
		return false;
	}
	bool operator>(const BigInt &b) const { return b < *this; }
	bool operator<=(const BigInt &b) const { return !(*this > b); }
	bool operator>=(const BigInt &b) const { return !(*this < b); }
	bool operator==(const BigInt &b) const { return nodes == b.nodes; }

	// 加法
	BigInt operator+(const BigInt &b) const
	{
		BigInt res = *this;
		int carry = 0;
		for (size_t i = 0; i < max(res.nodes.size(), b.nodes.size()) || carry; i++)
		{
			if (i == res.nodes.size())
				res.nodes.push_back(0);
			long long cur = (long long)res.nodes[i] + carry + (i < b.nodes.size() ? b.nodes[i] : 0);
			res.nodes[i] = cur % BASE;
			carry = cur / BASE;
		}
		return res;
	}

	// 减法 (假设 a >= b)
	BigInt operator-(const BigInt &b) const
	{
		BigInt res = *this;
		int carry = 0;
		for (size_t i = 0; i < b.nodes.size() || carry; i++)
		{
			long long cur = res.nodes[i] - carry - (i < b.nodes.size() ? b.nodes[i] : 0);
			carry = cur < 0;
			if (carry)
				cur += BASE;
			res.nodes[i] = cur;
		}
		res.trim();
		return res;
	}

	// 乘法
	BigInt operator*(const BigInt &b) const
	{
		BigInt res;
		res.nodes.resize(nodes.size() + b.nodes.size(), 0);
		for (size_t i = 0; i < nodes.size(); i++)
		{
			long long carry = 0;
			for (size_t j = 0; j < b.nodes.size() || carry; j++)
			{
				long long cur = res.nodes[i + j] + nodes[i] * 1LL * (j < b.nodes.size() ? b.nodes[j] : 0) + carry;
				res.nodes[i + j] = cur % BASE;
				carry = cur / BASE;
			}
		}
		res.trim();
		return res;
	}

	// 除法 (BigInt / long long)
	BigInt operator/(const long long v) const
	{
		BigInt res = *this;
		long long rem = 0;
		for (int i = res.nodes.size() - 1; i >= 0; i--)
		{
			long long cur = res.nodes[i] + rem * BASE;
			res.nodes[i] = cur / v;
			rem = cur % v;
		}
		res.trim();
		return res;
	}

	// 取模 (BigInt % long long)
	long long operator%(const long long v) const
	{
		long long m = 0;
		for (int i = nodes.size() - 1; i >= 0; i--)
			m = (m * BASE + nodes[i]) % v;
		return m;
	}

	// 输出流
	friend ostream &operator<<(ostream &out, const BigInt &a)
	{
		out << a.nodes.back();
		for (int i = a.nodes.size() - 2; i >= 0; i--)
			out << setfill('0') << setw(9) << a.nodes[i];
		return out;
	}

	// 输入流
	friend istream &operator>>(istream &in, BigInt &a)
	{
		string s;
		if (in >> s)
			a = BigInt(s);
		return in;
	}
};