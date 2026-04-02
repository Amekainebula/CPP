#include <bits/stdc++.h>
#define int long long
using namespace std;
const int N = 2005;
int a[N], p[N], l[N], r[N], w[N];
bool cmp(int x, int y) { return a[x] < a[y]; }
int f[N][N];
void dfs(int i, int k, bool bk, int now)
{
    if (i <= k)
    {
        int j = w[k]; // printf("%d\n",j);
        if (bk)
        {
            if (j <= now)
                now = r[now];
        }
        else
        {
            if (j >= now)
                now = l[now];
        }
        int r1 = r[l[j]], l1 = l[r[j]]; // tmp
        r[l[j]] = r[j];
        l[r[j]] = l[j];
        f[i][k - 1] = a[p[now]];
        dfs(i, k - 1, bk ^ 1, now /*目前中位数的位置*/);
        r[l[j]] = r1;
        l[r[j]] = l1;
    }
}
void make(int n)
{
    bool bk;
    if (n & 1)
        bk = false; // 奇数为false
    else
        bk = true;
    int now = (n + 1) / 2;
    f[1][n] = a[p[now]];
    dfs(1, n, bk, now);
    for (int i = 1; i < n; i++)
    {
        int j = w[i]; // printf("%d\n",j);
        if (bk)
        {
            if (j <= now)
                now = r[now];
        }
        else
        {
            if (j >= now)
                now = l[now];
        }
        r[l[j]] = r[j];
        l[r[j]] = l[j];
        bk ^= 1 /*奇偶性质*/; // 删数
        f[i + 1][n] = a[p[now]];
        dfs(i + 1, n, bk, now); // 处理中位数
    }
}
void clears(int n)
{
    for (int i = 1; i <= n; i++)
    {
        a[i] = 0;
        p[i] = 0;
        l[i] = 0;
        r[i] = 0;
        w[i] = 0;
        for (int j = 1; j <= n; j++)
        {
            f[i][j] = 0;
        }
    }
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    int t;
    cin >> t;
    while (t--)
    {
        int n;
        cin >> n;
        clears(n);
        for (int i = 1; i <= n; i++)
            cin >> a[i];
        for (int i = 1; i <= n; i++)
            p[i] = i;
        sort(p + 1, p + n + 1, cmp);
        for (int i = 1; i <= n; i++)
            w[p[i]] = i;
        for (int i = 1; i <= n; i++)
            l[i] = i - 1, r[i] = i + 1;
        make(n);
        int ans = 0;
        for (int i = 1; i <= n; i++)
        {
            for (int j = i; j <= n; j += 2)
            {
                ans += f[i][j] * i * j;
            }
        }
        cout << ans << endl;
    }
    return 0;
}