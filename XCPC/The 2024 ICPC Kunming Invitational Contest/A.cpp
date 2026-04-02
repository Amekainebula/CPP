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
struct Node
{
    int s, id;
    vector<int> p;
    int L, R, g;
    int miss;
};
void Murasame()
{
    int n, m;
    int k;
    read(n, m, k);
    vector<Node> a(n + 1);
    for (int i = 1; i <= n; i++)
    {
        a[i].p.assign(m + 1, 0);
        a[i].id = i;
        read(a[i].s);

        a[i].L = 0;
        a[i].miss = 0;
        for (int j = 1; j <= m; j++)
        {
            read(a[i].p[j]);
            if (a[i].p[j] == -1)
            {
                a[i].miss++;
            }
            else
            {
                a[i].L += a[i].p[j];
            }
        }
        a[i].R = a[i].L + 1LL * a[i].miss * k;
    }
    sort(a.begin() + 1, a.end(), [&](Node &A, Node &B)
         { return A.s < B.s; });
    int need = -1;
    for (int l = 1; l <= n;)
    {
        int r = l;
        while (r <= n && a[r].s == a[l].s)
            r++;
        int mx = need;
        for (int i = l; i < r; i++)
        {
            int w = need + 1;
            if (a[i].R < w)
            {
                wl("No");
                return;
            }
            a[i].g = max(a[i].L, w);
            mx = max(mx, a[i].g);
        }
        need = mx;
        l = r;
    }
    vector<vector<int>> ans(n + 1, vector<int>(m + 1, 0));
    for (int i = 1; i <= n; i++)
    {
        int tmp = a[i].g - a[i].L;
        for (int j = 1; j <= m; j++)
        {
            if (a[i].p[j] != -1)
            {
                ans[a[i].id][j] = a[i].p[j];
            }
            else
            {
                int x = min(k, tmp);
                ans[a[i].id][j] = x;
                tmp -= x;
            }
        }
    }

    wl("Yes");
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            wr(ans[i][j]);
        }
        wl();
    }
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