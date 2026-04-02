#include <bits/stdc++.h>
#define int long long
using namespace std;
bool H(string s)
{
    return s.size() == 1 && s[0] >= 'a' && s[0] <= 'z';
}
void solve()
{
    int ans = 0;
    string a, b, c, d, e, f;
    string A[6];
    for (int i = 1; i <= 5; i++)
        getline(cin, A[i]);
    for (int i = 0; i < A[2].size(); i++)
    {
        if (A[2][i] == '(')
        {
            int now = 1;
            while (A[2][i + now] != ',')
            {
                a += A[2][i + now];
                now++;
            }
            now++;
            while (A[2][i + now] != ',' && A[2][i + now] != ')')
            {
                b += A[2][i + now];
                now++;
            }
            if (A[2][i + now] == ',')
            {
                now++;
                while (A[2][i + now] != ')')
                {
                    c += A[2][i + now];
                    now++;
                }
            }
            break;
        };
    }
    for (int i = 0; i < A[3].size(); i++)
    {
        if (A[3][i] == '(')
        {
            int now = 1;
            while (A[3][i + now] != ',')
            {
                d += A[3][i + now];
                now++;
            }
            now++;
            while (A[3][i + now] != ',' && A[3][i + now] != ')')
            {
                e += A[3][i + now];
                now++;
            }
            if (A[3][i + now] == ',')
            {
                now++;
                while (A[3][i + now] != ')')
                {
                    f += A[3][i + now];
                    now++;
                }
            }
            break;
        }
    }
    int a1 = stoi(a), b1 = stoi(b), c1 = (c.empty() ? 1 : stoi(c));
    int d1 = H(d) ? -1e9 : stoi(d), e1 = H(e) ? -1e9 : stoi(e);
    int f1;
    if (f.empty())
        f1 = 1;
    else
    {
        if (H(f))
        {
            f1 = -1e9;
        }
        else
            f1 = stoi(f);
    }

    ans = 0;
    for (int i = a1; (c1 > 0) ? i < b1 : i > b1; i += c1)
    {

        int j = (d1 == -1e9 ? i : d1);
        if(!((f1 == -1e9 ? i : f1) > 0 ? j < (e1 == -1e9 ? i : e1)
                                  : j > (e1 == -1e9 ? i : e1)))
            continue;
        // cout << (e1 == -1e9 ? i : e1) << '\n';
        // cout << (d1 == -1e9 ? i : d1) << '\n';
        // cout << (f1 == -1e9 ? i : f1) << '\n';
        int n = (((e1 == -1e9 ? i : e1) - (d1 == -1e9 ? i : d1))) / (f1 == -1e9 ? i : f1);
        if ((((e1 == -1e9 ? i : e1) - (d1 == -1e9 ? i : d1))) % (f1 == -1e9 ? i : f1))
            n++;
       // cout << n << '\n';
        j = (d1 == -1e9 ? i : d1) + (f1 == -1e9 ? i : f1) * (n);
        // cout << "j:" << j << '\n';
        if(((f1 == -1e9 ? i : f1) > 0 ? j < (e1 == -1e9 ? i : e1)
                                  : j > (e1 == -1e9 ? i : e1)))
        {
            n++;
        }
        // cout << n << '\n';
        ans += n * (d1 == -1e9 ? i : d1) + (n - 1) * n / 2 * (f1 == -1e9 ? i : f1);
    }
    cout << ans << '\n';
}
signed main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    int _t = 1;
    // cin>>_t;
    while (_t--)
    {
        solve();
    }
    return 0;
}