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
template <typename Tp>
inline void read(vector<Tp> &x)
{
    for (int i = 1; i < x.size(); i++)
        read(x[i]);
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
template <typename Tp>
inline void write(const vector<Tp> &x)
{
    for (int i = 1; i < x.size(); i++)
        write(x[i]), i != x.size() - 1 && pc(' ');
    pc('\n');
}
template <typename T, typename... Tp>
inline void write(T x, Tp... y) { write(x), write(y...); }
template <typename Tp>
inline void wl(Tp x) { write(x), pc('\n'); }
inline void wl(const double x) { write(x), pc('\n'); }
inline void wl(const pair<int, double> x) { write(x), pc('\n'); }
inline void wl() { pc('\n'); }
template <typename T, typename... Tp>
inline void wl(T x, Tp... y) { wl(x), wl(y...); }
template <typename Tp>
inline void wr(Tp x) { write(x), pc(' '); }
inline void wr(const double x) { write(x), pc(' '); }
inline void wr(const pair<int, double> x) { write(x), pc(' '); }
inline void wr() { pc(' '); }
template <typename T, typename... Tp>
inline void wr(T x, Tp... y) { wr(x), wr(y...); }
const int MOD = 1e9 + 7;
const int mod = 998244353;
const int N = 1e6 + 6;
void Murasame()
{
    int n, a, b;
    read(n, a, b);
    int nn = n;
    double v1 = a / 7, v2 = b / 2, v3 = (a + b) / 8;
    int ans = 0;
    if (v1 > max(v2, v3))
    {
        ans += n / 7 * a;
        n %= 7;
    }
    else if (v2 >= max(v1, v3))
    {
        ans += n / 2 * b;
        n %= 2;
    }
    else
    {
        ans += n / 8 * (a + b);
        n %= 8;
    }
    if (n)
    {
        if (n >= 7)
        {
            if (v1 > v2)
            {
                ans += n / 7 * a;
                n %= 7;
            }
        }
        ans += n / 2 * b;
        n %= 2;
    }
    priority_queue<int> q;
    q.push(ans);
    q.push(nn / 7 * a + (nn % 7) / 2 * b);
    q.push(nn / 2 * b);
    q.push(nn / 8 * (a + b) + (nn % 8) / 7 * a + ((nn % 8) % 7) / 2 * b);
    q.push(nn / 8 * (a + b) + (nn % 8) / 2 * b);
    if (nn >= 2)
    {
        q.push((nn - 2) / 7 * a + ((nn - 2) % 7) / 2 * b + b);
        // q.push((nn - 2) / 2 * b + b);
        q.push((nn - 2) / 8 * (a + b) + ((nn - 2) % 8) / 7 * a + (((nn - 2) % 8) % 7) / 2 * b + b);
        q.push((nn - 2) / 8 * (a + b) + ((nn - 2) % 8) / 2 * b + b);
    }
    if (nn >= 7)
    {
        // q.push((nn - 7) / 7 * a + ((nn - 7) % 7) / 2 * b + a);
        q.push((nn - 7) / 2 * b + a);
        q.push((nn - 7) / 8 * (a + b) + ((nn - 7) % 8) / 7 * a + (((nn - 7) % 8) % 7) / 2 * b + a);
        q.push((nn - 7) / 8 * (a + b) + ((nn - 7) % 8) / 2 * b + a);
    }
    if (nn >= 8)
    {
        q.push((nn - 8) / 7 * a + ((nn - 8) % 7) / 2 * b + a + b);
        q.push((nn - 8) / 2 * b + a + b);
        // q.push((nn - 8) / 8 * (a + b) + ((nn - 8) % 8) / 7 * a + (((nn - 8) % 8) % 7) / 2 * b + a + b);
        // q.push((nn - 8) / 8 * (a + b) + ((nn - 8) % 8) / 2 * b + a + b);
    }
    // wl(q.top());
    vi aa = {0, n / 7, n / 8, (n + 1) / 8};
    for (int i = 0; i < 4; i++)
    {
        int has = aa[i];
        for (int x = has - 2; x <= has + 2; x++)//-2到+2
        {
            if (x < 0 || x * 7 > n)
                continue;
            int y11 = (n - 6 * x) / 2;
            if (y11 >= x)
            {
                q.push(x * a + y11 * b);
            }
            int y22 = min(x - 1, n - 7 * x);
            if (y22 >= 0)
            {
                q.push(x * a + y22 * b);
            }
        }
    }
    wl(q.top());
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

/*
一个长度为 n 的字符串，其每包含一个 qcjjkkt 子串，你获得 a 的快乐值，每包含一个 td 子串，获得b 的快乐值。问能够得到的最大快乐值是多少。
T<=1e5, n,a,b<=1e9
*/