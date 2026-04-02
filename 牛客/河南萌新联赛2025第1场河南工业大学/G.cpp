#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define vvi vector<vi>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
#define db(x) write(x), pc('\n')
#define dbb(x) write(x), pc(' ')
// #define endl endl << flush
#define endl '\n'
using namespace std;
const int MOD = 1e9 + 7;
const int mod = 998244353;
const int N = 5e5 + 6;
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
    register bool z = true;
    register char a = gc();
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
    register bool z = true;
    register double y = 0.1;
    register char a = gc();
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
    register char a = gc();
    for (; blank(a) && (a ^ -1); a = gc())
        ;
    for (; !blank(a) && (a ^ -1); a = gc())
        *x++ = a;
    *x = 0;
}
inline void read(string &x)
{
    x = "";
    register char a = gc();
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
    register int len = 0;
    register char tmp[64];
    for (; x; x /= 10)
        tmp[++len] = x % 10 + 48;
    while (len)
        pc(tmp[len--]);
}
inline void write(const double x)
{
    register int a = 6;
    register double b = x, c = b;
    if (b < 0)
        pc('-'), b = -b, c = -c;
    register double y = 5 * powl(10, -a - 1);
    b += y, c += y;
    register int len = 0;
    register char tmp[64];
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
    register int a = x.first;
    if (a < 7)
    {
        register double b = x.second, c = b;
        if (b < 0)
            pc('-'), b = -b, c = -c;
        register double y = 5 * powl(10, -a - 1);
        b += y, c += y;
        register int len = 0;
        register char tmp[64];
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

vi a(N), pre(N, -inf), suc(N, -inf);
void Murasame()
{
    int n;
    read(n);
    pre[n + 1] = -inf, pre[0] = -inf, suc[n + 1] = -inf, suc[0] = -inf;
    ff(i, 1, n)
    {
        read(a[i]);
        pre[i] = -inf;
        suc[i] = -inf;
    }
    int now = -inf;
    ff(i, 1, n)
    {
        pre[i] = max(a[i], now + a[i]);
        now = pre[i];
    }
    ff(i, 2, n)
    {
        pre[i] = max(pre[i - 1], pre[i]);
    }

    now = -inf;
    ffg(i, n, 1)
    {

        suc[i] = max(a[i], now + a[i]);
        now = suc[i];
    }
    ffg(i, n - 1, 1)
    {
        suc[i] = max(suc[i + 1], suc[i]);
    }

    int ans = -inf;
    ff(i, 1, n + 1)
    {
        ans = max(ans, pre[i - 1] + suc[i]);
    }
    db(ans);
    // now = 0;
    // int id2 = 0;
    // ffg(i, id, 1)
    // {
    //     now += a[i];
    //     if (now == mx)
    //     {
    //         id2 = i;
    //         break;
    //     }
    // }
    // int ans = -inf;
    // now = -inf;
    // ff(i, 1, id2 - 1)
    // {
    //     pre[i] = max(a[i], now + a[i]);
    //     ans = max(ans, pre[i]);
    //     now = pre[i];
    // }
    // now = -inf;
    // ff(i, id + 1, n)
    // {
    //     pre[i] = max(a[i], now + a[i]);
    //     ans = max(ans, pre[i]);
    //     now = pre[i];
    // }
    // cout << max(m1 + m2, ans + mx) << endl;
}
signed main()
{
    int _T = 1;
    //
    read(_T);
    while (_T--)
    {
        Murasame();
    }
    return 0;
}