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
const int N = 2e5 + 6;
int n, m, k;
string s;
int u, v;
void Murasame()
{
    read(n, m, k);
    read(s);
    s = '-' + s;
    vi g[n + 1], g2[n + 1];
    vc<vi> dis(3, vi(n + 1, LLONG_MAX));
    vi du(n + 1, 0), is(n + 1, 0);
    ff(i, 1, m)
    {
        read(u, v);
        du[u]++;
        g[u].pb(v);
        g2[v].pb(u);
    }
    set<int> temp;
    auto bfs1 = [&](int u, int type)
    {
        if (is[u] == 0 || is[u] == type)
        {
            is[u] = type;
        }
        else
        {
            is[u] = -1;
            temp.insert(u);
            return;
        }
        vi vis(n + 1, 0);
        queue<int> q;
        q.push(u);
        vis[u] = 1;
        while (!q.empty())
        {
            int u = q.front();
            q.pop();
            if (is[u] == 0 || is[u] == type)
            {
                is[u] = type;
            }
            else
            {
                is[u] = -1;
                temp.insert(u);
                continue;
            }
            for (int v : g2[u])
            {
                if (vis[v] == 0)
                {
                    vis[v] = 1;
                    q.push(v);
                }
            }
        }
    };
    ff(i, 1, n)
    {
        if (du[i] == 0)
        {
            bfs1(i, s[i] - 'A' + 1);
        }
    }
    auto clear = [&](int u)
    {
        vi vis(n + 1, 0);
        queue<int> q;
        q.push(u);
        vis[u] = 1;
        is[u] = 0;
        while (!q.empty())
        {
            int u = q.front();
            q.pop();
            vis[u] = 1;
            is[u] = 0;
            for (int v : g2[u])
            {
                if (is[v] != 0 && !vis[v])
                {
                    vis[v] = 1;
                    q.push(v);
                }
            }
        }
    };
    for (auto i : temp)
    {
        clear(i);
    }
    auto bfs2 = [&](int u, int type)
    {
        queue<int> q;
        q.push(u);
        dis[type][u] = 0;
        while (!q.empty())
        {
            int u = q.front();
            q.pop();
            for (int v : g2[u])
            {
                if (dis[type][v] > dis[type][u] + 1)
                {
                    dis[type][v] = dis[type][u] + 1;
                    q.push(v);
                }
            }
        }
    };
    ff(i, 1, n)
    {
        if (is[i] == 1)
        {
            bfs2(i, 1);
        }
        else if (is[i] == 2)
        {
            bfs2(i, 2);
        }
    }
    ff(i,1,2)
    {
        ff(j,1,n)
        {
            wr(dis[i][j]);
        }
        wr();
    }
    auto check = [&](int u, int type) -> int
    {
        queue<int> q;
        q.push(u);
        vi vis(n + 1, 0);
        int cnt = k;
        vis[u] = 1;
        while (!q.empty() && cnt)
        {
            int u = q.front();
            q.pop();
            cnt--;
            for (int v : g[u])
            {
                if (dis[type][v] == dis[type][u] + 1 && !vis[v])
                {
                    vis[v] = 1;
                    q.push(v);
                    break;
                }
            }
        }
        return q.front();
    };
    int ok = 1, tp;
    int uu = 1;
    while (1)
    {
        ok ^= 1;
        tp = ok + 1;
        if (dis[tp][uu] <= k)
        {
            wl(tp == 1 ? "Alice" : "Bob");
            return;
        }
        uu = check(uu, tp);
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