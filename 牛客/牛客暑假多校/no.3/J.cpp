#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define vvi vector<vi>
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
// #define endl endl << flush
#define endl '\n'
using namespace std;
#ifdef __linux__
#define gc getchar_unlocked
#define pc putchar_unlocked
#else
#define gc _getchar_nolock
#define pc _putchar_nolock
#endif
inline bool blank(const char x) { return !(x ^ 32) || !(x ^ 10) || !(x ^ 13) || !(x ^ 9); }
template <typename Tp>
inline void read(Tp &x)
{
	x = 0;
	bool z = true;
	char a = gc();
	for (; !isdigit(a); a = gc())
		if (a == '-')
			z = false;
	for (; isdigit(a); a = gc())
		x = (x << 1) + (x << 3) + (a ^ 48);
	x = (z ? x : ~x + 1);
}
inline void read(double &x)
{
	x = 0.0;
	bool z = true;
	double y = 0.1;
	char a = gc();
	for (; !isdigit(a); a = gc())
		if (a == '-')
			z = false;
	for (; isdigit(a); a = gc())
		x = x * 10 + (a ^ 48);
	if (a != '.')
		return x = z ? x : -x, void();
	for (a = gc(); isdigit(a); a = gc(), y /= 10)
		x += y * (a ^ 48);
	x = (z ? x : -x);
}
inline void read(char &x)
{
	for (x = gc(); blank(x) && (x ^ -1); x = gc())
		;
}
inline void read(char *x)
{
	char a = gc();
	for (; blank(a) && (a ^ -1); a = gc())
		;
	for (; !blank(a) && (a ^ -1); a = gc())
		*x++ = a;
	*x = 0;
}
inline void read(string &x)
{
	x = "";
	char a = gc();
	for (; blank(a) && (a ^ -1); a = gc())
		;
	for (; !blank(a) && (a ^ -1); a = gc())
		x += a;
}
template <typename T, typename... Tp>
inline void read(T &x, Tp &...y) { read(x), read(y...); }
template <typename Tp>
inline void write(Tp x)
{
	if (!x)
		return pc(48), void();
	if (x < 0)
		pc('-'), x = ~x + 1;
	int len = 0;
	char tmp[64];
	for (; x; x /= 10)
		tmp[++len] = x % 10 + 48;
	while (len)
		pc(tmp[len--]);
}
inline void write(const double x)
{
	int a = 6;
	double b = x, c = b;
	if (b < 0)
		pc('-'), b = -b, c = -c;
	double y = 5 * powl(10, -a - 1);
	b += y, c += y;
	int len = 0;
	char tmp[64];
	if (b < 1)
		pc(48);
	else
		for (; b >= 1; b /= 10)
			tmp[++len] = floor(b) - floor(b / 10) * 10 + 48;
	while (len)
		pc(tmp[len--]);
	pc('.');
	for (c *= 10; a; a--, c *= 10)
		pc(floor(c) - floor(c / 10) * 10 + 48);
}
inline void write(const pair<int, double> x)
{
	int a = x.first;
	if (a < 7)
	{
		double b = x.second, c = b;
		if (b < 0)
			pc('-'), b = -b, c = -c;
		double y = 5 * powl(10, -a - 1);
		b += y, c += y;
		int len = 0;
		char tmp[64];
		if (b < 1)
			pc(48);
		else
			for (; b >= 1; b /= 10)
				tmp[++len] = floor(b) - floor(b / 10) * 10 + 48;
		while (len)
			pc(tmp[len--]);
		a && (pc('.'));
		for (c *= 10; a; a--, c *= 10)
			pc(floor(c) - floor(c / 10) * 10 + 48);
	}
	else
		cout << fixed << setprecision(a) << x.second;
}
inline void write(const char x) { pc(x); }
inline void write(const bool x) { pc(x ? 49 : 48); }
inline void write(char *x) { fputs(x, stdout); }
inline void write(const char *x) { fputs(x, stdout); }
inline void write(const string &x) { fputs(x.c_str(), stdout); }
template <typename T, typename... Tp>
inline void write(T x, Tp... y) { write(x), write(y...); }
const int MOD = 1e9 + 7;
const int mod = 998244353;
const int N = 1e6 + 6;
void Murasame()
{
	int x, y;
	read(x, y);
	int sum = x + y;
	if (x == y)
	{
		write(1);
		pc('\n');
		return;
	}
	if (sum % 4 == 0)
	{
		int cnt = 0;
		int now1, now2;
		while (x != 0 && y != 0)
		{
			cnt++;
			now1 = min(x, y), now2 = max(x, y);
			if (x <= y)
			{
				y -= x;
				x += x;
			}
			else
			{
				x -= y;
				y += y;
			}
			if (now1 == min(x, y) && now2 == max(x, y)||cnt>=5000)
			{
				write(-1);
				pc('\n');
				return;
			}
		}
		write(cnt);
		pc('\n');
		return;
	}
	write(-1);
	pc('\n');
}
signed main()
{
	//	ios::sync_with_stdio(false);
	//	cin.tie(0);
	//	cout.tie(0);
	int _T = 1;
	//
	read(_T);
	while (_T--)
	{
		Murasame();
	}
	return 0;
}